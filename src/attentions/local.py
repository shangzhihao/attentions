from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention


class LocalSelfAttention(BaseSelfAttention):
    """Local Multi-Head Self-Attention implementation.
    
    This class implements local self-attention with multiple heads where each position 
    only attends to a local window of positions around it. This reduces computational 
    complexity from O(n²) to O(n*w) where w is the window size, while maintaining
    the representational power of multi-head attention.
    
    Args:
        d_model: Model dimension (must be divisible by num_heads)
        input_dim: Input feature dimension (default: None, uses d_model)
        window_size: Size of the local attention window (default: 128)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)
        temperature: Temperature scaling for attention scores (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        input_dim: Optional[int] = None,
        window_size: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__(d_model, input_dim, dropout, bias)
        
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.window_size = window_size
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
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, d_head]
        """
        batch_size, seq_len, d_model = x.shape
        # Reshape to [batch_size, seq_len, num_heads, d_head]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        # Transpose to [batch_size, num_heads, seq_len, d_head]
        return x.transpose(1, 2)
    
    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor back from multi-head attention format.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, d_head]
            
        Returns:
            Reshaped tensor [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_head = x.shape
        # Transpose to [batch_size, seq_len, num_heads, d_head]
        x = x.transpose(1, 2)
        # Reshape to [batch_size, seq_len, d_model]
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a local attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            
        Returns:
            Local attention mask [seq_len, seq_len] where 1 means attend, 0 means mask
        """
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        # Create local window mask
        half_window = self.window_size // 2
        
        for i in range(seq_len):
            # Define the local window boundaries
            start_idx = max(0, i - half_window)
            end_idx = min(seq_len, i + half_window + 1)
            
            # Allow attention within the local window
            mask[i, start_idx:end_idx] = True
        
        return mask
    
    def _apply_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply local multi-head attention with windowing.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, d_head]
            key: Key tensor [batch_size, num_heads, seq_len, d_head]
            value: Value tensor [batch_size, num_heads, seq_len, d_head]
            mask: Optional additional mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, num_heads, seq_len, d_head = query.shape
        
        # Create local attention mask
        local_mask = self._create_local_mask(seq_len, query.device)
        
        # Expand local mask for multiple heads: [batch_size, num_heads, seq_len, seq_len]
        local_mask_expanded = local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        # Combine with additional mask if provided
        if mask is not None:
            # Ensure mask has the right shape
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                pass  # Already in correct format
            
            # Apply both local mask and provided mask
            combined_mask = local_mask_expanded & mask.bool()
        else:
            combined_mask = local_mask_expanded
        
        # Apply scaled dot-product attention with the combined mask
        output, attention_weights = scaled_dot_product_attention(
            query, key, value,
            mask=combined_mask,
            dropout=self.dropout,
            temperature=self.temperature
        )
        
        return output, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of local multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask to combine with local mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)  # [batch_size, num_heads, seq_len, d_head]
        k = self._reshape_for_attention(k)  # [batch_size, num_heads, seq_len, d_head]
        v = self._reshape_for_attention(v)  # [batch_size, num_heads, seq_len, d_head]
        
        # Apply local attention with multiple heads
        attention_output, attention_weights = self._apply_local_attention(q, k, v, mask)
        
        # Reshape back to original format
        attention_output = self._reshape_from_attention(attention_output)
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Store attention weights for visualization (average across heads)
        # Shape: [batch_size, seq_len, seq_len]
        self.attention_weights = attention_weights.mean(dim=1).detach()
        
        return output, attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "temperature": self.temperature,
        })
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, window_size={self.window_size}, num_heads={self.num_heads}, d_head={self.d_head}, temperature={self.temperature}"