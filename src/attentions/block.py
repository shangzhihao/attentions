from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseSelfAttention, scaled_dot_product_attention
from .utils import reshape_for_attention, reshape_from_attention


class BlockSelfAttention(BaseSelfAttention):
    """Block-wise Multi-Head Self-Attention implementation.
    
    This class implements block-wise self-attention where the input sequence is
    divided into non-overlapping blocks, and attention is computed independently
    within each block. This reduces computational complexity from O(n²) to 
    O(b*(n/b)²) where b is the number of blocks and n is sequence length.
    
    Block attention is particularly effective for:
    - Long sequences where local patterns dominate
    - Memory-constrained environments
    - Hierarchical sequence modeling
    
    Args:
        d_model: Model dimension (must be divisible by num_heads)
        input_dim: Input feature dimension (default: None, uses d_model)
        block_size: Size of each attention block (default: 64)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear layers (default: False)
        temperature: Temperature scaling for attention scores (default: 1.0)
        overlap: Whether blocks should have overlap for boundary handling (default: False)
        rope: Whether to apply Rotary Position Embedding (default: False)
    """
    
    def __init__(
        self,
        d_model: int,
        input_dim: Optional[int] = None,
        block_size: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
        temperature: float = 1.0,
        overlap: bool = False,
        rope: bool = False,
    ):
        super().__init__(d_model, input_dim, dropout, bias, rope)
        
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature
        self.overlap = overlap
        
        # Linear projections for query, key, value, and output
        self.w_q = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_k = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_v = nn.Linear(self.input_dim, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _apply_block_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply block-wise multi-head attention.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, d_head]
            key: Key tensor [batch_size, num_heads, seq_len, d_head]
            value: Value tensor [batch_size, num_heads, seq_len, d_head]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, num_heads, seq_len, d_head = query.shape
        device = query.device
        
        # Get block indices
        block_starts, block_ends = self._create_block_indices(seq_len, self.block_size, self.overlap)
        num_blocks = len(block_starts)
        
        # Initialize output tensors
        output = torch.zeros_like(query)
        attention_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device, dtype=query.dtype)
        
        # Process each block
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            start_idx, end_idx = start.item(), end.item()
            block_len = end_idx - start_idx
            
            # Extract block data
            q_block = query[:, :, start_idx:end_idx, :]  # [batch_size, num_heads, block_len, d_head]
            k_block = key[:, :, start_idx:end_idx, :]    # [batch_size, num_heads, block_len, d_head]
            v_block = value[:, :, start_idx:end_idx, :]  # [batch_size, num_heads, block_len, d_head]
            
            # Extract block mask if provided
            block_mask = None
            if mask is not None:
                if mask.dim() == 2:  # [seq_len, seq_len]
                    block_mask = mask[start_idx:end_idx, start_idx:end_idx]
                    block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
                elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                    block_mask = mask[:, start_idx:end_idx, start_idx:end_idx]
                    block_mask = block_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
                elif mask.dim() == 4:  # [batch_size, num_heads, seq_len, seq_len]
                    block_mask = mask[:, :, start_idx:end_idx, start_idx:end_idx]
            
            # Apply attention within the block
            block_output, block_weights = scaled_dot_product_attention(
                q_block, k_block, v_block,
                mask=block_mask,
                dropout=self.dropout,
                temperature=self.temperature
            )
            
            # Store results
            if self.overlap and i > 0:
                # Handle overlapping blocks by averaging
                prev_end = block_starts[i-1].item() + (block_ends[i-1] - block_starts[i-1]).item()
                overlap_start = max(start_idx, prev_end - self.block_size + (self.block_size // 2))
                
                if overlap_start < end_idx:
                    # Average the overlapping region
                    overlap_len = end_idx - overlap_start
                    output_start_in_block = overlap_start - start_idx
                    output_end_in_block = output_start_in_block + overlap_len
                    
                    # Weight by position in block (linear interpolation)
                    alpha = torch.linspace(0.0, 1.0, int(overlap_len), device=device).view(1, 1, -1, 1)
                    
                    output[:, :, overlap_start:end_idx, :] = (
                        (1 - alpha) * output[:, :, overlap_start:end_idx, :] +
                        alpha * block_output[:, :, output_start_in_block:output_end_in_block, :]
                    )
                    
                    # Store non-overlapping part
                    if end_idx > overlap_start + overlap_len:
                        output[:, :, overlap_start + overlap_len:end_idx, :] = \
                            block_output[:, :, output_end_in_block:, :]
                else:
                    output[:, :, start_idx:end_idx, :] = block_output
            else:
                output[:, :, start_idx:end_idx, :] = block_output
            
            # Store attention weights (full matrix for visualization)
            attention_weights[:, :, start_idx:end_idx, start_idx:end_idx] = block_weights
        
        return output, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of block-wise multi-head self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, d_model]
            - attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle case where sequence is shorter than block size
        if seq_len <= self.block_size:
            # Fall back to regular attention for short sequences
            q = self.w_q(x)
            k = self.w_k(x)
            v = self.w_v(x)
            
            # Reshape for multi-head attention
            q = reshape_for_attention(q, self.num_heads, self.d_head)
            k = reshape_for_attention(k, self.num_heads, self.d_head)
            v = reshape_for_attention(v, self.num_heads, self.d_head)
            
            # Apply RoPE if enabled
            if self.rope:
                q, k = self.apply_rope(q, k)
            
            # Apply regular attention
            attention_output, attention_weights = scaled_dot_product_attention(
                q, k, v, mask=mask, dropout=self.dropout, temperature=self.temperature
            )
            
            # Reshape back
            attention_output = reshape_from_attention(attention_output, self.d_model)
            
        else:
            # Apply linear projections
            q = self.w_q(x)  # [batch_size, seq_len, d_model]
            k = self.w_k(x)  # [batch_size, seq_len, d_model]
            v = self.w_v(x)  # [batch_size, seq_len, d_model]
            
            # Reshape for multi-head attention
            q = reshape_for_attention(q, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
            k = reshape_for_attention(k, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
            v = reshape_for_attention(v, self.num_heads, self.d_head)  # [batch_size, num_heads, seq_len, d_head]
            
            # Apply RoPE if enabled
            if self.rope:
                q, k = self.apply_rope(q, k)
            
            # Apply block attention
            attention_output, attention_weights = self._apply_block_attention(q, k, v, mask)
            
            # Reshape back to original format
            attention_output = reshape_from_attention(attention_output, self.d_model)
        
        # Apply output projection
        output = self.w_o(attention_output)
        
        # Store attention weights for visualization (average across heads)
        self.attention_weights = attention_weights.mean(dim=1).detach()
        
        return output, attention_weights
    
    def _create_block_indices(
        self,
        seq_len: int,
        block_size: int,
        overlap: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create block start and end indices for block-wise attention.
        
        Args:
            seq_len: Length of the input sequence
            block_size: Size of each attention block
            overlap: Whether blocks should have 50% overlap
            
        Returns:
            Tuple of (block_starts, block_ends) tensors
        """
        if overlap:
            # Create overlapping blocks with 50% overlap
            step = max(1, block_size // 2)
            starts = list(range(0, seq_len, step))
            
            # Ensure we don't go beyond sequence length
            block_starts = []
            block_ends = []
            
            for start in starts:
                end = min(start + block_size, seq_len)
                if start < seq_len:  # Only add valid blocks
                    block_starts.append(start)
                    block_ends.append(end)
                if end >= seq_len:  # Stop if we've reached the end
                    break
        else:
            # Create non-overlapping blocks
            num_blocks = (seq_len + block_size - 1) // block_size  # Ceiling division
            block_starts = [i * block_size for i in range(num_blocks)]
            block_ends = [min((i + 1) * block_size, seq_len) for i in range(num_blocks)]
        
        return torch.tensor(block_starts), torch.tensor(block_ends)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "block_size": self.block_size,
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "temperature": self.temperature,
            "overlap": self.overlap,
        })
        return config
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        base_repr = super().extra_repr()
        return f"{base_repr}, block_size={self.block_size}, num_heads={self.num_heads}, d_head={self.d_head}, temperature={self.temperature}, overlap={self.overlap}"