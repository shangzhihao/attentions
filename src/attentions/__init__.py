"""Attentions: A PyTorch library for attention mechanisms.

This library provides clean, efficient implementations of various self-attention
mechanisms used in transformer models and deep learning architectures.
"""

from .base import (
    BaseSelfAttention,
    scaled_dot_product_attention,
)
from .vanilla import VanillaSelfAttention
from .mhsa import MultiHeadSelfAttention
from .local import LocalSelfAttention

__version__ = "0.1.0"

__all__ = [
    "BaseSelfAttention",
    "VanillaSelfAttention",
    "MultiHeadSelfAttention",
    "LocalSelfAttention",
    "scaled_dot_product_attention",
]
