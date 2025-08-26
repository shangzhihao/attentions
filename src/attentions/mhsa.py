from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import math

from .base import BaseSelfAttention, scaled_dot_product_attention


class MultiHeadSelfAttention(BaseSelfAttention):
    """Multi-Head Self-Attention implementation.
    
    This class implements the multi-head self-attention mechanism where the input
    is linearly projected into multiple heads, attention is computed in parallel
    for each head, and the results are concatenated and projected back.
    
    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)
        temperature: Temperature scaling for attention scores (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        temperature: float = 1.0,
    ):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        super().__init__(d_model, input_dim, dropout, bias)
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections for query, key, value, and output
        # Using single linear layers and reshaping for efficiency
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
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len] or broadcastable
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, d_model]
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = self._reshape_for_attention(q)  # [batch_size, num_heads, seq_len, d_head]
        k = self._reshape_for_attention(k)  # [batch_size, num_heads, seq_len, d_head]
        v = self._reshape_for_attention(v)  # [batch_size, num_heads, seq_len, d_head]
        
        # Expand mask for multiple heads if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                # Expand to [batch_size, num_heads, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif mask.dim() == 2:  # [seq_len, seq_len]
                # Expand to [batch_size, num_heads, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        
        # Compute scaled dot-product attention for each head
        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, temperature=self.temperature
        )
        
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
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "temperature": self.temperature,
        })
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, num_heads={self.num_heads}, d_head={self.d_head}, temperature={self.temperature}"