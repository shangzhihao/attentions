from typing import Any

import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention
from .masks import create_local_mask
from .utils import reshape_for_attention, reshape_from_attention


class LongformerSelfAttention(BaseSelfAttention):
    """Longformer-style sliding window self-attention with optional global tokens.

    Implements the Longformer attention pattern:
    - Sliding window local attention of size ``window_size`` around each token
    - Optional global tokens which can attend to all tokens and be attended by all

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        input_dim: Input feature dimension (default: None, uses d_model)
        window_size: Sliding window size (default: 512)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        rope: Whether to apply Rotary Position Embedding (default: False)

    Forward inputs:
        x: Tensor of shape [batch, seq, input_dim]
        mask: Optional attention mask. Supports:
            - bool: [seq, seq], [batch, seq, seq], or [batch, num_heads, seq, seq]
            - additive: same shapes, where masked positions are large negative
        global_attention: Optional bool mask indicating global tokens.
            Shapes: [seq] or [batch, seq]. True means global.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        input_dim: int | None = None,
        window_size: int = 512,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
        rope: bool = False,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        super().__init__(d_model, input_dim, dropout, bias, rope)

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.window_size = window_size
        self.temperature = temperature

        # Projections
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _build_longformer_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        base_mask: torch.Tensor | None,
        global_attention: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Create a combined Longformer attention mask expanded to heads.

        Returns a boolean mask of shape [batch, num_heads, seq, seq] where
        True means attend and False means masked out. If both a base mask and
        global mask are provided, the result respects both (logical AND with
        the base mask).
        """
        # Start with sliding-window local mask
        local = create_local_mask(seq_len, self.window_size, device)
        long_mask = local.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        # Apply global attention if provided
        if global_attention is not None:
            if global_attention.dim() == 1:
                g = global_attention.unsqueeze(0).expand(batch_size, -1)
            elif global_attention.dim() == 2:
                g = global_attention
            else:
                raise ValueError("global_attention must be [seq] or [batch, seq] boolean mask")
            if g.dtype != torch.bool:
                g = g.bool()

            # For each batch, for global indices j:
            # - allow all i to attend to j (set column j True)
            # - allow j to attend to all (set row j True)
            # Vectorized updates
            # Columns: broadcast g over row dimension
            # long_mask[b, h, :, j] = True where g[b, j] is True
            g_cols = g.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, seq_len, -1)
            long_mask = long_mask | g_cols

            # Rows: broadcast g over column dimension
            # long_mask[b, h, j, :] = True where g[b, j] is True
            g_rows = g.view(batch_size, 1, seq_len, 1).expand(-1, self.num_heads, -1, seq_len)
            long_mask = long_mask | g_rows

        if base_mask is None:
            return long_mask

        # Combine with user-provided mask
        if base_mask.dim() == 2:
            base_exp = base_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
        elif base_mask.dim() == 3:
            base_exp = base_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        elif base_mask.dim() == 4:
            base_exp = base_mask
        else:
            raise ValueError("mask must be 2D/3D/4D tensor")

        if base_exp.dtype == torch.bool:
            return long_mask & base_exp
        else:
            # Convert longformer mask to additive, then add
            add_long = torch.where(long_mask, 0.0, float("-inf")).to(base_exp.dtype).to(base_exp.device)
            return add_long + base_exp

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        global_attention: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape to heads
        q = reshape_for_attention(q, self.num_heads, self.d_head)
        k = reshape_for_attention(k, self.num_heads, self.d_head)
        v = reshape_for_attention(v, self.num_heads, self.d_head)

        # RoPE if enabled
        if self.rope:
            q, k = self.apply_rope(q, k)

        # Build Longformer mask and move/cast to match q/k dtype/device if additive
        combined_mask = self._build_longformer_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            base_mask=mask,
            global_attention=global_attention,
        )

        # Compute attention
        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=combined_mask, dropout=self.dropout, temperature=self.temperature
        )

        # Back to [batch, seq, d_model]
        attn_out = reshape_from_attention(attn_out, self.d_model)
        output = self.w_o(attn_out)

        # Store averaged attention weights for convenience
        self.attention_weights = attn_weights.mean(dim=1).detach()

        return output, attn_weights

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "d_head": self.d_head,
                "window_size": self.window_size,
                "temperature": self.temperature,
            }
        )
        return config

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, num_heads={self.num_heads}, d_head={self.d_head}, window_size={self.window_size}, temperature={self.temperature}"

