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
from .longformer import LongformerSelfAttention
from .lsh import LSHSelfAttention
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

__all__ = (
    "AlibiSelfAttention",
    "BaseSelfAttention",
    "BlockSelfAttention",
    "DilatedSelfAttention",
    "GroupedSelfAttention",
    "LSHSelfAttention",
    "LinearSelfAttention",
    "LocalSelfAttention",
    "LongformerSelfAttention",
    "MultiHeadSelfAttention",
    "VanillaSelfAttention",
    "combine_masks",
    "create_block_mask",
    "create_causal_mask",
    "create_dilated_mask",
    "create_local_mask",
    "create_padding_mask",
    "expand_mask_for_heads",
    "scaled_dot_product_attention",
)
