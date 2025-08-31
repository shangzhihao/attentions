"""Attentions: A PyTorch library for attention mechanisms.

This library provides clean, efficient implementations of various self-attention
mechanisms used in transformer models and deep learning architectures.
"""

from .alibi import AlibiSelfAttention
from .base import (
    BaseSelfAttention,
    scaled_dot_product_attention,
)
from .block import BlockSelfAttention
from .dilation import DilatedSelfAttention
from .group import GroupedSelfAttention
from .linear import LinearSelfAttention
from .local import LocalSelfAttention
from .masks import (
    combine_masks,
    create_block_mask,
    create_causal_mask,
    create_dilated_mask,
    create_local_mask,
    create_padding_mask,
    expand_mask_for_heads,
)
from .mhsa import MultiHeadSelfAttention
from .vanilla import VanillaSelfAttention

__version__ = "0.1.0"

__all__ = [
    # Base classes and core functions
    "BaseSelfAttention",
    "scaled_dot_product_attention",
    # Attention mechanisms
    "VanillaSelfAttention",
    "MultiHeadSelfAttention",
    "LocalSelfAttention",
    "GroupedSelfAttention",
    "DilatedSelfAttention",
    "LinearSelfAttention",
    "BlockSelfAttention",
    "AlibiSelfAttention",
    # Mask utilities
    "create_causal_mask",
    "create_padding_mask",
    "create_local_mask",
    "create_dilated_mask",
    "create_block_mask",
    "combine_masks",
    "expand_mask_for_heads",
]
