"""Test cases for LocalSelfAttention implementation."""

import math
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from src.attentions.local import LocalSelfAttention


class TestLocalSelfAttention:
    """Test cases for LocalSelfAttention class."""
    
    def test_initialization(self):
        """Test proper initialization of LocalSelfAttention."""
        d_model = 64
        window_size = 32
        num_heads = 8
        dropout = 0.1
        temperature = 1.5
        
        attention = LocalSelfAttention(
            d_model=d_model,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            temperature=temperature
        )
        
        # Test basic attributes
        assert attention.d_model == d_model
        assert attention.window_size == window_size
        assert attention.num_heads == num_heads
        assert attention.d_head == d_model // num_heads
        assert attention.dropout_prob == dropout
        assert attention.bias is True
        assert attention.temperature == temperature
        
        # Test linear layers
        assert isinstance(attention.w_q, nn.Linear)
        assert isinstance(attention.w_k, nn.Linear)
        assert isinstance(attention.w_v, nn.Linear)
        assert isinstance(attention.w_o, nn.Linear)
    
    def test_initialization_validation(self):
        """Test window_size and num_heads validation."""
        # Should work
        LocalSelfAttention(d_model=64, window_size=1, num_heads=8)
        LocalSelfAttention(d_model=128, window_size=128, num_heads=16)
        
        # Should fail - invalid window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            LocalSelfAttention(d_model=64, window_size=0, num_heads=8)
        
        with pytest.raises(ValueError, match="window_size must be positive"):
            LocalSelfAttention(d_model=64, window_size=-1, num_heads=8)
        
        # Should fail - d_model not divisible by num_heads
        with pytest.raises(ValueError, match="d_model .* must be divisible by num_heads"):
            LocalSelfAttention(d_model=65, window_size=8, num_heads=8)
        
        with pytest.raises(ValueError, match="d_model .* must be divisible by num_heads"):
            LocalSelfAttention(d_model=100, window_size=8, num_heads=7)
    
    def test_local_mask_creation(self):
        """Test local attention mask creation."""
        attention = LocalSelfAttention(d_model=32, window_size=4, num_heads=4)
        seq_len = 8
        device = torch.device('cpu')
        
        mask = attention._create_local_mask(seq_len, device)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check that each position can attend to its local window
        for i in range(seq_len):
            half_window = attention.window_size // 2
            start_idx = max(0, i - half_window)
            end_idx = min(seq_len, i + half_window + 1)
            
            # Within window should be True
            for j in range(start_idx, end_idx):
                assert mask[i, j] == True
            
            # Outside window should be False
            for j in range(seq_len):
                if j < start_idx or j >= end_idx:
                    assert mask[i, j] == False
    
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        batch_size, seq_len, d_model = 2, 16, 64
        window_size = 8
        num_heads = 8
        
        attention = LocalSelfAttention(d_model=d_model, window_size=window_size, num_heads=num_heads)
        attention.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(x)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check that attention is only applied within local windows for each head
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(seq_len):
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(seq_len, i + half_window + 1)
                    
                    # Within window should have non-zero attention
                    window_weights = attention_weights[b, h, i, start_idx:end_idx]
                    assert torch.allclose(window_weights.sum(), torch.tensor(1.0), atol=1e-6)
                    
                    # Outside window should have zero attention
                    if start_idx > 0:
                        assert torch.allclose(attention_weights[b, h, i, :start_idx], torch.zeros(start_idx))
                    if end_idx < seq_len:
                        assert torch.allclose(attention_weights[b, h, i, end_idx:], torch.zeros(seq_len - end_idx))
    
    def test_forward_pass_with_additional_mask(self):
        """Test forward pass with additional mask combined with local mask."""
        batch_size, seq_len, d_model = 1, 8, 32
        window_size = 6
        num_heads = 4
        
        attention = LocalSelfAttention(d_model=d_model, window_size=window_size, num_heads=num_heads)
        attention.eval()
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        
        output, attention_weights = attention(x, mask=causal_mask)
        
        # Check shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check that both local and causal masks are applied for each head
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Get expected mask value (both local and causal)
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(seq_len, i + half_window + 1)
                    
                    local_allowed = start_idx <= j < end_idx
                    causal_allowed = j <= i
                    
                    if local_allowed and causal_allowed:
                        # Should have some attention weight
                        continue  # Can't test exact values due to softmax
                    else:
                        # Should have zero or near-zero attention
                        assert attention_weights[0, h, i, j] < 1e-5
    
    def test_attention_weights_storage(self):
        """Test that attention weights are properly stored and retrievable."""
        attention = LocalSelfAttention(d_model=32, window_size=8, num_heads=4)
        x = torch.randn(1, 10, 32)
        
        # Test before forward pass
        with pytest.raises(RuntimeError):
            attention.get_attention_weights()
        
        # Test after forward pass
        output, weights = attention(x)
        stored_weights = attention.get_attention_weights()
        
        # Stored weights should be averaged across heads
        expected_weights = weights.mean(dim=1).detach()
        assert torch.allclose(stored_weights, expected_weights)
        assert stored_weights.requires_grad is False
        assert stored_weights.shape == (1, 10, 10)
        
        # Original weights should have multi-head shape
        assert weights.shape == (1, 4, 10, 10)
    
    def test_different_head_counts(self):
        """Test attention with different numbers of heads."""
        d_model = 64
        seq_len = 16
        window_size = 8
        x = torch.randn(1, seq_len, d_model)
        
        for num_heads in [1, 2, 4, 8, 16]:
            attention = LocalSelfAttention(d_model=d_model, window_size=window_size, num_heads=num_heads)
            attention.eval()
            
            output, attention_weights = attention(x)
            
            assert output.shape == (1, seq_len, d_model)
            assert attention_weights.shape == (1, num_heads, seq_len, seq_len)
            assert attention.d_head == d_model // num_heads
            
            # Check that each head has proper local windowing
            for h in range(num_heads):
                for i in range(seq_len):
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(seq_len, i + half_window + 1)
                    
                    # Count non-zero attention weights for this head
                    non_zero_count = (attention_weights[0, h, i] > 1e-6).sum().item()
                    expected_window = end_idx - start_idx
                    assert non_zero_count == expected_window
    
    def test_different_window_sizes(self):
        """Test attention with different window sizes."""
        d_model = 32
        seq_len = 16
        num_heads = 8
        x = torch.randn(1, seq_len, d_model)
        
        for window_size in [2, 4, 8, 16, 32]:
            attention = LocalSelfAttention(d_model=d_model, window_size=window_size, num_heads=num_heads)
            attention.eval()
            
            output, attention_weights = attention(x)
            
            assert output.shape == (1, seq_len, d_model)
            assert attention_weights.shape == (1, num_heads, seq_len, seq_len)
            
            # Check sparsity pattern for each head
            for h in range(num_heads):
                for i in range(seq_len):
                    half_window = window_size // 2
                    start_idx = max(0, i - half_window)
                    end_idx = min(seq_len, i + half_window + 1)
                    
                    # Count non-zero attention weights for this head
                    non_zero_count = (attention_weights[0, h, i] > 1e-6).sum().item()
                    expected_window = end_idx - start_idx
                    assert non_zero_count == expected_window
    