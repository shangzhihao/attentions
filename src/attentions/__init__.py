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
from .group import GroupedSelfAttention
from .dilation import DilatedSelfAttention
from .linear import LinearSelfAttention
from .block import BlockSelfAttention

__version__ = "0.1.0"

__all__ = [
    "BaseSelfAttention",
    "VanillaSelfAttention",
    "MultiHeadSelfAttention",
    "LocalSelfAttention",
    "GroupedSelfAttention",
    "DilatedSelfAttention",
    "LinearSelfAttention",
    "BlockSelfAttention",
    "scaled_dot_product_attention",
]
