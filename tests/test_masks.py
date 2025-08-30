"""Tests for attention mask utilities."""

import pytest
import torch

from attentions.masks import (
    create_causal_mask,
    create_padding_mask, 
    create_local_mask,
    create_dilated_mask,
    create_block_mask,
    combine_masks,
    expand_mask_for_heads,
)


def test_create_causal_mask():
    """Test causal mask creation."""
    seq_len = 4
    device = torch.device("cpu")
    
    mask = create_causal_mask(seq_len, device)
    
    # Check shape and dtype
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool
    assert mask.device == device
    
    # Check that it's lower triangular
    expected = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    assert torch.equal(mask, expected)
    
    # Check specific pattern
    assert mask[0, 0] == True
    assert mask[0, 1] == False
    assert mask[1, 0] == True
    assert mask[1, 1] == True
    assert mask[2, 3] == False
    assert mask[3, 2] == True


def test_create_padding_mask():
    """Test padding mask creation."""
    batch_size = 3
    max_len = 5
    seq_lengths = torch.tensor([3, 5, 2])
    device = torch.device("cpu")
    
    mask = create_padding_mask(seq_lengths, max_len, device)
    
    # Check shape and dtype
    assert mask.shape == (batch_size, max_len)
    assert mask.dtype == torch.bool
    assert mask.device == device
    
    # Check padding pattern
    expected = torch.tensor([
        [True, True, True, False, False],    # seq_len=3
        [True, True, True, True, True],      # seq_len=5
        [True, True, False, False, False]    # seq_len=2
    ])
    assert torch.equal(mask, expected)


def test_create_padding_mask_auto_max_len():
    """Test padding mask with automatic max_len detection."""
    seq_lengths = torch.tensor([2, 4, 1])
    
    mask = create_padding_mask(seq_lengths)
    
    # Should use max(seq_lengths) = 4
    assert mask.shape == (3, 4)
    
    expected = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False]
    ])
    assert torch.equal(mask, expected)


def test_create_local_mask():
    """Test local attention mask creation."""
    seq_len = 6
    window_size = 3
    device = torch.device("cpu")
    
    mask = create_local_mask(seq_len, window_size, device)
    
    # Check shape and dtype
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool
    assert mask.device == device
    
    # Check local window pattern
    half_window = window_size // 2  # = 1
    
    # Position 0: can attend to [0, 1]
    assert mask[0, 0] == True
    assert mask[0, 1] == True
    assert mask[0, 2] == False
    
    # Position 2: can attend to [1, 2, 3]
    assert mask[2, 0] == False
    assert mask[2, 1] == True
    assert mask[2, 2] == True
    assert mask[2, 3] == True
    assert mask[2, 4] == False
    
    # Position 5: can attend to [4, 5]
    assert mask[5, 3] == False
    assert mask[5, 4] == True
    assert mask[5, 5] == True


def test_create_dilated_mask():
    """Test dilated attention mask creation."""
    seq_len = 8
    dilation_rate = 2
    device = torch.device("cpu")
    
    mask = create_dilated_mask(seq_len, dilation_rate, device)
    
    # Check shape and dtype
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool
    assert mask.device == device
    
    # Check self-attention (diagonal should be True)
    for i in range(seq_len):
        assert mask[i, i] == True
    
    # Check dilation pattern for position 2
    # Should attend to: 0 (backward), 2 (self), 4, 6 (forward)
    expected_row_2 = torch.zeros(seq_len, dtype=torch.bool)
    expected_row_2[[0, 2, 4, 6]] = True
    assert torch.equal(mask[2], expected_row_2)
    
    # Check dilation pattern for position 3
    # Should attend to: 1 (backward), 3 (self), 5, 7 (forward)
    expected_row_3 = torch.zeros(seq_len, dtype=torch.bool)
    expected_row_3[[1, 3, 5, 7]] = True
    assert torch.equal(mask[3], expected_row_3)


def test_create_block_mask():
    """Test block attention mask creation."""
    seq_len = 9
    block_size = 3
    device = torch.device("cpu")
    
    mask = create_block_mask(seq_len, block_size, device)
    
    # Check shape and dtype
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool
    assert mask.device == device
    
    # Expected block pattern:
    # Block 0: [0,1,2] x [0,1,2]
    # Block 1: [3,4,5] x [3,4,5] 
    # Block 2: [6,7,8] x [6,7,8]
    
    # Within first block
    assert mask[0, 0] == True
    assert mask[0, 2] == True
    assert mask[1, 2] == True
    
    # Across blocks should be False
    assert mask[0, 3] == False
    assert mask[2, 6] == False
    assert mask[3, 8] == False
    
    # Within second block
    assert mask[3, 3] == True
    assert mask[4, 5] == True
    
    # Within third block
    assert mask[6, 8] == True
    assert mask[8, 6] == True


def test_combine_masks():
    """Test mask combination with logical AND."""
    size = 4
    device = torch.device("cpu")
    
    # Create two test masks
    mask1 = torch.tril(torch.ones(size, size, dtype=torch.bool))  # Causal
    mask2 = create_local_mask(size, 2, device)  # Local
    
    combined = combine_masks(mask1, mask2)
    
    # Check shape and dtype
    assert combined.shape == (size, size)
    assert combined.dtype == torch.bool
    
    # Combined should be True only where both are True
    expected = mask1 & mask2
    assert torch.equal(combined, expected)


def test_combine_masks_multiple():
    """Test combining multiple masks."""
    size = 3
    device = torch.device("cpu")
    
    mask1 = torch.ones(size, size, dtype=torch.bool)
    mask2 = torch.tril(torch.ones(size, size, dtype=torch.bool))
    mask3 = torch.eye(size, dtype=torch.bool)  # Only diagonal
    
    combined = combine_masks(mask1, mask2, mask3)
    
    # Should only have diagonal elements
    expected = torch.eye(size, dtype=torch.bool)
    assert torch.equal(combined, expected)


def test_combine_masks_empty():
    """Test combine_masks with no arguments."""
    with pytest.raises(ValueError, match="At least one mask must be provided"):
        combine_masks()


def test_combine_masks_shape_mismatch():
    """Test combine_masks with incompatible shapes."""
    mask1 = torch.ones(3, 3, dtype=torch.bool)
    mask2 = torch.ones(4, 4, dtype=torch.bool)
    
    with pytest.raises(ValueError, match="Mask shape mismatch"):
        combine_masks(mask1, mask2)


def test_expand_mask_for_heads_2d():
    """Test mask expansion from 2D to 4D."""
    seq_len = 4
    batch_size = 2
    num_heads = 3
    
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    expanded = expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    assert expanded.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Check that mask is properly replicated
    for b in range(batch_size):
        for h in range(num_heads):
            assert torch.equal(expanded[b, h], mask)


def test_expand_mask_for_heads_3d():
    """Test mask expansion from 3D to 4D."""
    seq_len = 3
    batch_size = 2
    num_heads = 4
    
    mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool))
    expanded = expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    assert expanded.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Check that mask is properly replicated across heads
    for b in range(batch_size):
        for h in range(num_heads):
            assert torch.equal(expanded[b, h], mask[b])


def test_expand_mask_for_heads_4d():
    """Test mask expansion from 4D (already correct shape)."""
    seq_len = 3
    batch_size = 2
    num_heads = 2
    
    mask = torch.tril(torch.ones(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool))
    expanded = expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    # Should return the same mask
    assert torch.equal(expanded, mask)


def test_expand_mask_for_heads_none():
    """Test mask expansion with None input."""
    result = expand_mask_for_heads(None, 2, 4, 3)
    assert result is None


def test_expand_mask_for_heads_invalid_dimensions():
    """Test mask expansion with invalid dimensions."""
    seq_len = 4
    batch_size = 2
    num_heads = 3
    
    # Wrong 2D shape
    mask = torch.ones(3, 3, dtype=torch.bool)  # seq_len should be 4
    with pytest.raises(ValueError, match="2D mask shape"):
        expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    # Wrong 3D shape
    mask = torch.ones(3, seq_len, seq_len, dtype=torch.bool)  # batch_size should be 2
    with pytest.raises(ValueError, match="3D mask shape"):
        expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    # Wrong 4D shape
    mask = torch.ones(batch_size, 2, seq_len, seq_len, dtype=torch.bool)  # num_heads should be 3
    with pytest.raises(ValueError, match="4D mask shape"):
        expand_mask_for_heads(mask, batch_size, num_heads, seq_len)
    
    # Unsupported dimensions
    mask = torch.ones(2, 3, 4, 5, 6, dtype=torch.bool)  # 5D
    with pytest.raises(ValueError, match="Unsupported mask dimensions"):
        expand_mask_for_heads(mask, batch_size, num_heads, seq_len)


def test_mask_device_consistency():
    """Test that masks are created on the correct device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    seq_len = 4
    
    causal = create_causal_mask(seq_len, device)
    local = create_local_mask(seq_len, 2, device)
    dilated = create_dilated_mask(seq_len, 2, device)
    block = create_block_mask(seq_len, 2, device)
    
    assert causal.device == device
    assert local.device == device
    assert dilated.device == device
    assert block.device == device


def test_mask_boolean_dtype():
    """Test that all masks have boolean dtype."""
    device = torch.device("cpu")
    seq_len = 4
    
    causal = create_causal_mask(seq_len, device)
    local = create_local_mask(seq_len, 2, device)
    dilated = create_dilated_mask(seq_len, 2, device)
    block = create_block_mask(seq_len, 2, device)
    
    assert causal.dtype == torch.bool
    assert local.dtype == torch.bool
    assert dilated.dtype == torch.bool
    assert block.dtype == torch.bool