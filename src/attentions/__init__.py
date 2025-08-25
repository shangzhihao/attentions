"""Attentions: A PyTorch library for attention mechanisms.

This library provides clean, efficient implementations of various self-attention
mechanisms used in transformer models and deep learning architectures.
"""

from .attention import (
    BaseAttention,
    SelfAttention,
    VanillaSelfAttention,
    scaled_dot_product_attention,
)

__version__ = "0.1.0"

__all__ = [
    "BaseAttention",
    "SelfAttention", 
    "VanillaSelfAttention",
    "scaled_dot_product_attention",
]
