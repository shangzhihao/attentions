"""Regression tests for shared attention primitives."""

import torch

from attzoo.base import scaled_dot_product_attention


def test_scaled_dot_product_attention_all_masked_row_is_finite() -> None:
    """Fully masked rows should not produce NaNs."""
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)

    mask = torch.tensor(
        [
            [True, False, False],
            [False, False, False],
            [True, True, False],
        ]
    )

    out, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    assert not torch.isnan(weights).any()
    assert not torch.isnan(out).any()
    assert torch.allclose(weights[0, 1], torch.zeros(3))
    assert torch.allclose(out[0, 1], torch.zeros(8))
