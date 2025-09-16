from typing import Any

import math
import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention
from .utils import reshape_for_attention, reshape_from_attention
from .masks import expand_mask_for_heads


class LSHSelfAttention(BaseSelfAttention):
    """LSH (Reformer-style) self-attention.

    This module approximates full attention by hashing tokens into buckets using
    random projections, then computing attention only within each bucket. Multiple
    independent hash rounds can be used and averaged to improve recall.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        bucket_size: Target bucket size; number of buckets ~= ceil(seq_len / bucket_size)
        num_hashes: Number of independent hashing rounds to average (default: 1)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        input_dim: int | None = None,
        bucket_size: int = 64,
        num_hashes: int = 1,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        if num_hashes <= 0:
            raise ValueError(f"num_hashes must be positive, got {num_hashes}")

        super().__init__(d_model, input_dim, dropout, bias, rope)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.bucket_size = bucket_size
        self.num_hashes = num_hashes
        self.temperature = temperature

        # Projections
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _compute_num_buckets(self, seq_len: int) -> int:
        # At least 1 bucket; prefer smaller number of buckets when seq is short
        return max(1, math.ceil(seq_len / self.bucket_size))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for LSH self-attention.

        Args:
            x: Input tensor [batch, seq, input_dim]
            mask: Optional attention mask. Supports same formats as MHSA:
                  [seq, seq], [batch, seq, seq], or [batch, heads, seq, seq],
                  with either boolean (True=attend) or additive (-inf for masked).

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, seq, d_model]
            - attention_weights: [batch, num_heads, seq, seq] (sparse; zeros outside buckets)
        """
        batch_size, seq_len, _ = x.shape
        H, Dh = self.num_heads, self.d_head

        # Projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # To heads
        q = reshape_for_attention(q, H, Dh)
        k = reshape_for_attention(k, H, Dh)
        v = reshape_for_attention(v, H, Dh)

        # RoPE
        if self.rope:
            q, k = self.apply_rope(q, k)

        # Expand mask to [B, H, S, S] if provided
        expanded_mask = expand_mask_for_heads(mask, batch_size, H, seq_len)

        # Prepare accumulators
        out_accum = torch.zeros_like(q)
        attn_weights_accum = q.new_zeros((batch_size, H, seq_len, seq_len))

        n_buckets = self._compute_num_buckets(seq_len)

        # Perform num_hashes rounds and average
        for _ in range(self.num_hashes):
            # Random projections per head: [H, Dh, n_buckets]
            # Non-trainable, re-sampled each forward for diversity
            proj = torch.randn(H, Dh, n_buckets, device=x.device, dtype=x.dtype)

            # Scores to buckets: [B, H, S, n_buckets]
            scores = torch.einsum("bhld,hdm->bhlm", q, proj)  # cosine-like LSH by random directions
            bucket_ids = scores.argmax(dim=-1)  # [B, H, S]

            # Compute attention within each bucket
            for b in range(batch_size):
                for h in range(H):
                    # Optional base mask for this (b, h)
                    base_mask_bh = None
                    if expanded_mask is not None:
                        base_mask_bh = expanded_mask[b, h]

                    # Group indices by bucket id
                    ids_bh = bucket_ids[b, h]
                    for bucket in range(n_buckets):
                        idx = (ids_bh == bucket).nonzero(as_tuple=False).squeeze(-1)
                        if idx.numel() == 0:
                            continue

                        q_g = q[b, h, idx]  # [Sg, Dh]
                        k_g = k[b, h, idx]  # [Sg, Dh]
                        v_g = v[b, h, idx]  # [Sg, Dh]

                        # Build group mask if available
                        group_mask = None
                        if base_mask_bh is not None:
                            group_mask = base_mask_bh.index_select(0, idx).index_select(1, idx)

                        # Compute attention inside the group
                        out_g, attn_g = scaled_dot_product_attention(
                            q_g.unsqueeze(0),
                            k_g.unsqueeze(0),
                            v_g.unsqueeze(0),
                            mask=group_mask.unsqueeze(0) if group_mask is not None else None,
                            dropout=self.dropout,
                            temperature=self.temperature,
                        )
                        out_g = out_g.squeeze(0)  # [Sg, Dh]
                        attn_g = attn_g.squeeze(0)  # [Sg, Sg]

                        # Scatter-add outputs back to positions
                        out_accum[b, h, idx] = out_accum[b, h, idx] + out_g

                        # Place attention weights into the global matrix (sparse)
                        # Build (row, col) index grids for the group positions
                        ii = idx.view(-1, 1).expand(attn_g.size(0), attn_g.size(1))
                        jj = idx.view(1, -1).expand(attn_g.size(0), attn_g.size(1))
                        # Accumulate since multiple hashes may write to same entries
                        attn_weights_accum[b, h].index_put_((ii, jj), attn_g, accumulate=True)

        # Average across hash rounds
        out_accum = out_accum / float(self.num_hashes)
        attn_weights_accum = attn_weights_accum / float(self.num_hashes)

        # Back to [B, S, D]
        out_cat = reshape_from_attention(out_accum, self.d_model)
        output = self.w_o(out_cat)

        # Store averaged attention weights for convenience (mean over heads)
        self.attention_weights = attn_weights_accum.mean(dim=1).detach()

        return output, attn_weights_accum

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_head": self.d_head,
                "bucket_size": self.bucket_size,
                "num_hashes": self.num_hashes,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, num_heads={self.num_heads}, d_head={self.d_head}, "
            f"bucket_size={self.bucket_size}, num_hashes={self.num_hashes}, "
            f"temperature={self.temperature}"
        )
