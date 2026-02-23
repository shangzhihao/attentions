"""Regression tests for CombinedAttention."""

import torch

from attzoo.combined import CombinedAttention
from attzoo.vanilla import VanillaSelfAttention


def test_combined_attention_uses_wrapped_input_dim_by_default() -> None:
    """Default gate input_dim should follow wrapped attention modules."""
    d_model = 16
    input_dim = 8
    seq_len = 5
    batch_size = 2

    attn_a = VanillaSelfAttention(d_model=d_model, input_dim=input_dim, dropout=0.0)
    attn_b = VanillaSelfAttention(d_model=d_model, input_dim=input_dim, dropout=0.0)
    combined = CombinedAttention(attn_a, attn_b)
    combined.eval()

    x = torch.randn(batch_size, seq_len, input_dim)
    output, weights = combined(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, seq_len, seq_len)
