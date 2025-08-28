"""Attention mechanisms for transformer models.

This module provides a comprehensive implementation of various attention mechanisms
including self-attention, multi-head attention, and cross-attention patterns.
All implementations follow the architectural patterns established for modularity
and extensibility.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)
    
    # Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            scores = scores + mask
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
    

class BaseSelfAttention(nn.Module, ABC):
    """Base class for self-attention mechanisms.
    
    Self-attention computes relationships within a single input sequence by
    using the same tensor for queries, keys, and values. This allows each
    position to attend to all positions in the same sequence.
    
    Args:
        d_model: Model dimension for attention computation
        input_dim: Input feature dimension (default: None, uses d_model)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)
    """
    
    def __init__(
        self, 
        d_model: int, 
        input_dim: Optional[int] = None,
        dropout: float = 0.1, 
        bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim if input_dim is not None else d_model
        self.dropout_prob = dropout
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        
        # Initialize attention weights storage
        self.attention_weights: Optional[torch.Tensor] = None
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for self-attention.
        
        Args:
            x: Value tensor (should be same as query for self-attention)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # This is an abstract implementation that should be overridden
        raise NotImplementedError("Subclasses must implement the forward method")
    def get_attention_weights(self) -> torch.Tensor:
        """Get attention weights from the most recent forward pass.
        
        Returns:
            Attention weights tensor
            
        Raises:
            RuntimeError: If no forward pass has been performed
        """
        if self.attention_weights is None:
            raise RuntimeError(
                "No attention weights available. Perform a forward pass first."
            )
        return self.attention_weights
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.
        Returns:
            Configuration dictionary
        """
        return {
            "d_model": self.d_model,
            "input_dim": self.input_dim,
            "dropout": self.dropout_prob,
            "bias": self.bias,
        }
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"d_model={self.d_model}, input_dim={self.input_dim}, dropout={self.dropout_prob}, bias={self.bias}"


