from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSelfAttention


class LinearSelfAttention(BaseSelfAttention):
    """Linear self-attention implementation with O(n) complexity.
    
    This class implements linear attention using feature maps to approximate
    the softmax operation, achieving linear complexity instead of quadratic.
    Uses ELU activation with added constant for feature mapping.
    
    Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    by Katharopoulos et al. (2020)
    
    Args:
        d_model: Model dimension (dimension for Q, K, V projections and output)
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        feature_dim: Dimension of feature mapping (default: None, uses d_model)
        eps: Small constant for numerical stability (default: 1e-6)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """
    
    def __init__(
        self,
        d_model: int,
        input_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        feature_dim: Optional[int] = None,
        eps: float = 1e-6,
        rope: bool = False,
    ):
        super().__init__(d_model, input_dim, dropout, bias, rope)
        
        self.feature_dim = feature_dim if feature_dim is not None else d_model
        self.eps = eps
        
        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, self.feature_dim, bias=bias)
        self.w_k = nn.Linear(self.input_dim, self.feature_dim, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature mapping function to approximate softmax.
        
        Uses ELU activation with added constant as in the original paper.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature mapped tensor
        """
        return F.elu(x) + 1.0
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of linear self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask (currently not supported for linear attention)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights) where output is [batch_size, seq_len, d_model]
            Note: attention_weights are computed for compatibility but are approximate
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply linear projections
        q = self.w_q(x)  # [batch_size, seq_len, feature_dim]
        k = self.w_k(x)  # [batch_size, seq_len, feature_dim]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # Apply RoPE if enabled (only to q and k)
        if self.rope:
            # Add head dimension for RoPE compatibility
            q_with_head = q.unsqueeze(1)  # [batch_size, 1, seq_len, feature_dim]
            k_with_head = k.unsqueeze(1)  # [batch_size, 1, seq_len, feature_dim]
            q_with_head, k_with_head = self.apply_rope(q_with_head, k_with_head)
            q = q_with_head.squeeze(1)  # [batch_size, seq_len, feature_dim]
            k = k_with_head.squeeze(1)  # [batch_size, seq_len, feature_dim]
        
        # Apply feature mapping
        q_prime = self._feature_map(q)  # [batch_size, seq_len, feature_dim]
        k_prime = self._feature_map(k)  # [batch_size, seq_len, feature_dim]
        
        # Handle masking for linear attention
        if mask is not None:
            if mask.dtype == torch.bool:
                # Convert boolean mask to float
                mask = mask.float()
            
            # Handle different mask shapes
            if mask.dim() == 2 and mask.size(0) == seq_len and mask.size(1) == seq_len:
                # Attention mask [seq_len, seq_len] -> convert to sequence mask
                # Take diagonal to get sequence validity
                mask = mask.diagonal().unsqueeze(0).expand(batch_size, -1)
            elif mask.dim() == 3 and mask.size(1) == mask.size(2):
                # Attention mask [batch_size, seq_len, seq_len] -> sequence mask
                mask = mask.diagonal(dim1=1, dim2=2)
            elif mask.dim() == 2 and mask.size(0) == batch_size:
                # Already sequence mask [batch_size, seq_len]
                pass
            else:
                # Assume it's a sequence mask, ensure proper shape
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0).expand(batch_size, -1)
            
            # Apply mask to keys and values
            mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            k_prime = k_prime * mask
            v = v * mask
        
        # Linear attention computation: O(n) complexity
        # Compute K^T V: [batch_size, feature_dim, d_model]
        kv = torch.einsum('bsf,bsd->bfd', k_prime, v)
        
        # Compute normalization: sum of keys for each feature dimension
        k_sum = k_prime.sum(dim=1)  # [batch_size, feature_dim]
        
        # Compute output: Q * (K^T V) / (Q * K^T 1)
        numerator = torch.einsum('bsf,bfd->bsd', q_prime, kv)  # [batch_size, seq_len, d_model]
        denominator = torch.einsum('bsf,bf->bs', q_prime, k_sum).unsqueeze(-1) + self.eps
        
        attention_output = numerator / denominator
        
        # Apply dropout
        attention_output = self.dropout(attention_output)
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Compute approximate attention weights for compatibility
        # This is not exact for linear attention but provides a reasonable approximation
        with torch.no_grad():
            q_norm = q_prime / (q_prime.sum(dim=-1, keepdim=True) + self.eps)
            k_norm = k_prime / (k_prime.sum(dim=-1, keepdim=True) + self.eps)
            attention_weights = torch.bmm(q_norm, k_norm.transpose(-2, -1))
            
            # Normalize attention weights
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + self.eps)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights.detach()
        
        return output, attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "eps": self.eps,
        })
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, feature_dim={self.feature_dim}, eps={self.eps}"