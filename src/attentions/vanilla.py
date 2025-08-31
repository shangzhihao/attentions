from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention
class VanillaSelfAttention(BaseSelfAttention):
    """Vanilla self-attention implementation.
    
    This class implements the basic scaled dot-product self-attention mechanism.
    
    Args:
        d_model: Model dimension (dimension for Q, K, V projections and output)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """
    
    def __init__(
        self,
        d_model: int,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        super().__init__(d_model, input_dim, dropout, bias, rope)
        self.temperature = temperature
        
        # Create standard linear projection layers
        self.w_q, self.w_k, self.w_v, self.w_o = self._create_projection_layers(bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of vanilla self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights) where output is [batch_size, seq_len, d_model]
        """
        # Apply linear projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Apply RoPE if enabled
        if self.rope:
            q, k = self.apply_rope(q, k)
        
        # Compute scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, temperature=self.temperature
        )
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights.detach()
        
        return output, attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config["temperature"] = self.temperature
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, temperature={self.temperature}"