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
    """Compute scaled dot-product attention.
    
    This function implements the core attention mechanism described in
    "Attention Is All You Need" (Vaswani et al., 2017) with additional
    support for temperature scaling and masking.
    
    Mathematical formulation:
        Attention(Q, K, V) = softmax(QK^T / √(d_k * temperature)) * V
    
    Args:
        query: Query tensor with shape [..., seq_len_q, d_k]
        key: Key tensor with shape [..., seq_len_k, d_k]
        value: Value tensor with shape [..., seq_len_k, d_v]
        mask: Optional mask tensor with shape [..., seq_len_q, seq_len_k]
        dropout: Optional dropout layer applied to attention weights
        temperature: Temperature scaling factor (default: 1.0)
        
    Returns:
        Tuple containing:
            - output: Attention output tensor with shape [..., seq_len_q, d_v]
            - attention_weights: Attention weights with shape [..., seq_len_q, seq_len_k]
    """
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


class BaseAttention(nn.Module, ABC):
    """Abstract base class for all attention mechanisms.
    
    This class defines the common interface that all attention mechanisms
    must implement, ensuring consistency across different attention types
    and enabling polymorphic usage in transformer architectures.
    
    Args:
        d_model: The model dimension
        dropout: Dropout probability for regularization
        bias: Whether to include bias terms in linear layers
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.dropout_prob = dropout
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        
        # Initialize attention weights storage
        self.attention_weights: Optional[torch.Tensor] = None
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the attention mechanism.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass
    
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
            "dropout": self.dropout_prob,
            "bias": self.bias,
        }
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f"d_model={self.d_model}, dropout={self.dropout_prob}, bias={self.bias}"


class SelfAttention(BaseAttention):
    """Base class for self-attention mechanisms.
    
    Self-attention computes relationships within a single input sequence by
    using the same tensor for queries, keys, and values. This allows each
    position to attend to all positions in the same sequence.
    """
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for self-attention.
        
        For self-attention, query, key, and value should be the same tensor.
        This base implementation can be overridden by specific self-attention
        implementations.
        
        Args:
            query: Query tensor
            key: Key tensor (should be same as query for self-attention)
            value: Value tensor (should be same as query for self-attention)
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # This is an abstract implementation that should be overridden
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def forward_self(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for self-attention with single input tensor.
        
        Args:
            x: Input tensor to use as query, key, and value
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        return self.forward(x, x, x, mask, **kwargs)


class VanillaSelfAttention(SelfAttention):
    """Vanilla self-attention implementation.
    
    This class implements the basic scaled dot-product self-attention mechanism
    as described in "Attention Is All You Need". It includes linear projections
    for query, key, value, and output transformations.
    
    Args:
        d_model: Model dimension
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: True)
        temperature: Temperature scaling for attention scores (default: 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        bias: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__(d_model, dropout, bias)
        self.temperature = temperature
        
        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize linear layer weights using Xavier uniform initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of vanilla self-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Apply linear projections
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
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
