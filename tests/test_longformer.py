"""Tests for LongformerSelfAttention (sliding window + global tokens)."""

import pytest
import torch

from attentions.longformer import LongformerSelfAttention


def test_longformer_forward_basic_shapes():
    batch_size, seq_len, d_model = 2, 12, 48
    num_heads, window_size = 4, 6
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    out, weights = attn(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_longformer_window_pattern_and_global_tokens():
    """Verify local window sparsity and global token behavior."""
    batch_size, seq_len, d_model = 1, 10, 40
    num_heads, window_size = 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Mark index 6 as global
    g_idx = 6
    global_mask = torch.zeros(seq_len, dtype=torch.bool)
    global_mask[g_idx] = True

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x, global_attention=global_mask)

    half = window_size // 2

    # 1) Local window: ensure masked positions outside window are zero for non-global columns
    for head in range(num_heads):
        for i in range(seq_len):
            # Skip global row: global token can attend to all positions
            if i == g_idx:
                continue
            start = max(0, i - half)
            end = min(seq_len, i + half + 1)
            # Indices strictly outside local window AND not global should be zero
            before = list(range(0, start))
            after = list(range(end, seq_len))
            for j in before + after:
                if j != g_idx:  # global col exempt from local mask
                    assert w[0, head, i, j] < 1e-7

    # 2) Global column: every position can attend to the global token's column
    for head in range(num_heads):
        # Some probability mass should be assigned to column g_idx for every i
        assert (w[0, head, :, g_idx] > 1e-7).all()

    # 3) Global row: the global token attends to all positions (row fully allowed)
    for head in range(num_heads):
        assert (w[0, head, g_idx, :] > 1e-7).all()


def test_longformer_with_causal_mask_and_globals():
    """Causal base mask should still constrain future positions even for globals."""
    batch_size, seq_len, d_model = 1, 8, 32
    num_heads, window_size = 2, 4
    x = torch.randn(batch_size, seq_len, d_model)

    g = 5  # global position
    global_mask = torch.zeros(seq_len, dtype=torch.bool)
    global_mask[g] = True

    causal = torch.tril(torch.ones(seq_len, seq_len)).bool()

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x, mask=causal, global_attention=global_mask)

    # For i < g, attending to future global position g must be masked by causal
    for head in range(num_heads):
        for i in range(g):
            assert w[0, head, i, g] < 1e-7

    # For i >= g, attending to global position g is allowed
    for head in range(num_heads):
        for i in range(g, seq_len):
            assert w[0, head, i, g] > 1e-7


def test_longformer_additive_mask_support():
    """Additive mask (-inf for masked) should zero-out those positions."""
    batch_size, seq_len, d_model = 1, 7, 28
    num_heads, window_size = 2, 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Mask the last column for everyone via additive mask
    additive = torch.zeros(seq_len, seq_len)
    additive[:, -1] = float("-inf")

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x, mask=additive)

    # Last column must be zero for all heads and all i
    for head in range(num_heads):
        assert (w[0, head, :, -1] < 1e-7).all()


def test_longformer_window_larger_than_sequence_behaves_full():
    batch_size, seq_len, d_model = 1, 5, 20
    num_heads, window_size = 2, 64  # window >> seq_len
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x)

    # All entries should be unmasked (positive after softmax)
    assert (w > 1e-7).all()


def test_longformer_single_element_window_self_only():
    batch_size, seq_len, d_model = 1, 6, 24
    num_heads, window_size = 3, 1
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()
    _, w = attn(x)

    # Only diagonal should be non-zero when no globals
    for head in range(num_heads):
        for i in range(seq_len):
            expected = torch.zeros(seq_len)
            expected[i] = 1.0
            assert torch.allclose(w[0, head, i], expected, atol=1e-6)


def test_longformer_temperature_scaling():
    batch_size, seq_len, d_model = 1, 8, 32
    num_heads, window_size = 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    low = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, temperature=0.1
    )
    high = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, temperature=10.0
    )
    low.eval(); high.eval()

    # Align parameters for fair comparison
    with torch.no_grad():
        for p_low, p_high in zip(low.parameters(), high.parameters()):
            p_high.copy_(p_low)

    _, w_low = low(x)
    _, w_high = high(x)

    # Low temperature should yield sharper (higher max) attention per head
    for head in range(num_heads):
        assert w_low[0, head].max() >= w_high[0, head].max()


def test_longformer_rope_toggle_and_stored_weights():
    batch_size, seq_len, d_model = 1, 10, 40
    num_heads, window_size = 5, 6
    x = torch.randn(batch_size, seq_len, d_model)

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size, rope=True
    )
    attn.eval()
    out, w = attn(x)

    # Shapes
    assert out.shape == (batch_size, seq_len, d_model)
    assert w.shape == (batch_size, num_heads, seq_len, seq_len)

    # Stored weights are mean over heads
    stored = attn.get_attention_weights()
    assert torch.allclose(stored, w.mean(dim=1), atol=1e-6)
    assert stored.shape == (batch_size, seq_len, seq_len)


def test_longformer_global_mask_batch_and_vector_forms():
    batch_size, seq_len, d_model = 2, 9, 36
    num_heads, window_size = 3, 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Vector form (same globals for all in batch)
    v = torch.zeros(seq_len, dtype=torch.bool)
    v[2] = True

    # Batch form (different for sample 0 and 1)
    b = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    b[0, 2] = True
    b[1, 7] = True

    attn = LongformerSelfAttention(
        d_model=d_model, num_heads=num_heads, window_size=window_size
    )
    attn.eval()

    # Vector form
    _, w_vec = attn(x, global_attention=v)

    # Batch form
    _, w_batch = attn(x, global_attention=b)

    # For sample 0, column 2 must be >0 everywhere in both cases
    for head in range(num_heads):
        assert (w_vec[0, head, :, 2] > 1e-7).all()
        assert (w_batch[0, head, :, 2] > 1e-7).all()

    # For sample 1, vector form has no global at 7; batch form does
    for head in range(num_heads):
        # Batch form should enable col 7 for all i
        assert (w_batch[1, head, :, 7] > 1e-7).all()
        # Vector form: column 7 behaves as local-only for non-window positions
        # Find an i far from 7
        i_far = 0
        half = window_size // 2
        outside_window = not (max(0, i_far - half) <= 7 < min(seq_len, i_far + half + 1))
        assert outside_window
        assert w_vec[1, head, i_far, 7] < 1e-7
