"""Attention mask utilities for various attention mechanisms.

This module provides centralized mask creation functions that can be reused
across different attention mechanism implementations. All mask functions follow
the convention where True/1 means "attend" and False/0 means "mask out".
"""

from typing import Optional, Tuple

import torch


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        
    Returns:
        Causal mask tensor [seq_len, seq_len] where True means attend, False means mask
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def create_padding_mask(
    seq_lengths: torch.Tensor, 
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create a padding mask for sequences with different lengths.
    
    Args:
        seq_lengths: Tensor of actual sequence lengths [batch_size]
        max_len: Maximum sequence length (default: max of seq_lengths)
        device: Device to create the mask on (default: same as seq_lengths)
        
    Returns:
        Padding mask tensor [batch_size, max_len] where True means valid, False means padded
    """
    if device is None:
        device = seq_lengths.device
    if max_len is None:
        max_len = seq_lengths.max().item()
    
    batch_size = seq_lengths.size(0)
    
    # Create position indices
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask: position < seq_length
    mask = positions < seq_lengths.unsqueeze(1)
    
    return mask


def create_local_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Create a local attention mask for windowed attention.
    
    Args:
        seq_len: Sequence length
        window_size: Size of the local attention window
        device: Device to create the mask on
        
    Returns:
        Local attention mask [seq_len, seq_len] where True means attend, False means mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    
    # Create local window mask
    half_window = window_size // 2
    
    for i in range(seq_len):
        # Define the local window boundaries
        start_idx = max(0, i - half_window)
        end_idx = min(seq_len, i + half_window + 1)
        
        # Allow attention within the local window
        mask[i, start_idx:end_idx] = True
    
    return mask


def create_dilated_mask(seq_len: int, dilation_rate: int, device: torch.device) -> torch.Tensor:
    """Create a dilated attention mask for sparse attention patterns.
    
    Args:
        seq_len: Sequence length
        dilation_rate: Dilation rate for attention pattern
        device: Device to create the mask on
        
    Returns:
        Dilated mask tensor [seq_len, seq_len] where True means attend, False means mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    
    for i in range(seq_len):
        # Always attend to self
        mask[i, i] = True
        
        # Attend to positions at dilation intervals
        # Forward direction
        for j in range(i + dilation_rate, seq_len, dilation_rate):
            mask[i, j] = True
        
        # Backward direction
        for j in range(i - dilation_rate, -1, -dilation_rate):
            mask[i, j] = True
    
    return mask


def create_block_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Create a block attention mask for block-wise attention.
    
    Args:
        seq_len: Sequence length
        block_size: Size of each attention block
        device: Device to create the mask on
        
    Returns:
        Block mask tensor [seq_len, seq_len] where True means attend, False means mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    
    # Create block diagonal pattern
    num_blocks = (seq_len + block_size - 1) // block_size  # Ceiling division
    
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, seq_len)
        
        # Allow attention within each block
        mask[start_idx:end_idx, start_idx:end_idx] = True
    
    return mask


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """Combine multiple attention masks using logical AND.
    
    Args:
        *masks: Variable number of attention masks to combine
        
    Returns:
        Combined mask where all input masks must allow attention
        
    Raises:
        ValueError: If masks have incompatible shapes
    """
    if not masks:
        raise ValueError("At least one mask must be provided")
    
    result = masks[0].clone()
    
    for mask in masks[1:]:
        if mask.shape != result.shape:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {result.shape}")
        result = result & mask.bool()
    
    return result


def expand_mask_for_heads(
    mask: torch.Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int
) -> torch.Tensor:
    """Expand attention mask for multi-head attention.
    
    Args:
        mask: Input mask tensor
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        
    Returns:
        Expanded mask tensor [batch_size, num_heads, seq_len, seq_len]
        
    Raises:
        ValueError: If mask dimensions are incompatible
    """
    if mask is None:
        return None
    
    # Handle different input mask formats
    if mask.dim() == 2:  # [seq_len, seq_len]
        if mask.size(0) != seq_len or mask.size(1) != seq_len:
            raise ValueError(f"2D mask shape {mask.shape} incompatible with seq_len {seq_len}")
        expanded = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
        if mask.size(0) != batch_size or mask.size(1) != seq_len or mask.size(2) != seq_len:
            raise ValueError(f"3D mask shape {mask.shape} incompatible with dimensions ({batch_size}, {seq_len}, {seq_len})")
        expanded = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
        if mask.shape != (batch_size, num_heads, seq_len, seq_len):
            raise ValueError(f"4D mask shape {mask.shape} incompatible with expected ({batch_size}, {num_heads}, {seq_len}, {seq_len})")
        expanded = mask
    else:
        raise ValueError(f"Unsupported mask dimensions: {mask.dim()}")
    
    return expanded