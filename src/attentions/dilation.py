from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention
from .utils import reshape_for_attention, reshape_from_attention


class DilatedSelfAttention(BaseSelfAttention):
    """Dilated Self-Attention implementation.
    
    This class implements dilated self-attention where attention is computed
    with a dilation pattern, similar to dilated convolutions. This allows
    the model to attend to positions at regular intervals, which can be
    useful for capturing long-range dependencies efficiently while reducing
    computational complexity.
    
    Args:
        d_model: Model dimension for attention computation
        dilation_rate: Dilation rate for attention pattern (default: 2)
        num_heads: Number of attention heads (default: 1)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)
        temperature: Temperature scaling for attention scores (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        dilation_rate: int = 2,
        num_heads: int = 1,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
    ):
        if dilation_rate < 1:
            raise ValueError(f"dilation_rate must be >= 1, got {dilation_rate}")
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        super().__init__(d_model, input_dim, dropout, bias)
        
        self.dilation_rate = dilation_rate
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize linear layer weights using Xavier uniform initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _create_dilated_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create dilated attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            
        Returns:
            Dilated mask tensor [seq_len, seq_len] where 1 means attend, 0 means mask
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            # Always attend to self
            mask[i, i] = True
            
            # Attend to positions at dilation intervals
            # Forward direction
            for j in range(i + self.dilation_rate, seq_len, self.dilation_rate):
                mask[i, j] = True
            
            # Backward direction
            for j in range(i - self.dilation_rate, -1, -self.dilation_rate):
                mask[i, j] = True
        
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of dilated self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len] or broadcastable
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention if num_heads > 1
        if self.num_heads > 1:
            q = reshape_for_attention(q, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
            k = reshape_for_attention(k, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
            v = reshape_for_attention(v, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
        else:
            q = q.unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
            k = k.unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
            v = v.unsqueeze(1)  # [batch_size, 1, seq_len, d_model]
        
        # Create dilated attention mask
        dilated_mask = self._create_dilated_mask(seq_len, x.device)
        
        # Combine with user-provided mask if given
        if mask is not None:
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Convert boolean mask to match dilated_mask format if needed
            if mask.dtype == torch.bool:
                combined_mask = dilated_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1) & mask
            else:
                # For additive masks, convert dilated_mask to additive format
                dilated_additive = torch.where(dilated_mask, 0.0, float('-inf'))
                dilated_additive = dilated_additive.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                combined_mask = dilated_additive + mask
        else:
            # Use only dilated mask
            combined_mask = dilated_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        # Compute scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=combined_mask, dropout=self.dropout, temperature=self.temperature
        )
        
        # Reshape back to original format
        if self.num_heads > 1:
            attention_output = reshape_from_attention(attention_output, self.d_model)
        else:
            attention_output = attention_output.squeeze(1)  # Remove head dimension
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Store attention weights for visualization (average across heads)
        # Shape: [batch_size, seq_len, seq_len]
        if self.num_heads > 1:
            self.attention_weights = attention_weights.mean(dim=1).detach()
        else:
            self.attention_weights = attention_weights.squeeze(1).detach()
        
        return output, attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "dilation_rate": self.dilation_rate,
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "temperature": self.temperature,
        })
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, dilation_rate={self.dilation_rate}, "
            f"num_heads={self.num_heads}, d_head={self.d_head}, "
            f"temperature={self.temperature}"
        )