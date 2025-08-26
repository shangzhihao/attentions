"""Test cases for MultiHeadSelfAttention implementation."""

import math
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from src.attentions.mhsa import MultiHeadSelfAttention


class TestMultiHeadSelfAttention:
    """Test cases for MultiHeadSelfAttention class."""
    
    def test_initialization(self):
        """Test proper initialization of MultiHeadSelfAttention."""
        d_model = 64
        num_heads = 8
        dropout = 0.1
        temperature = 1.5
        
        attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            temperature=temperature
        )
        
        # Test basic attributes
        assert attention.d_model == d_model
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
        
        # Test layer dimensions
        assert attention.w_q.in_features == d_model
        assert attention.w_q.out_features == d_model
    
    def test_initialization_dimension_validation(self):
        """Test that d_model must be divisible by num_heads."""
        # Should work
        MultiHeadSelfAttention(d_model=64, num_heads=8)
        MultiHeadSelfAttention(d_model=128, num_heads=16)
        
        # Should fail
        with pytest.raises(ValueError, match="d_model .* must be divisible by num_heads"):
            MultiHeadSelfAttention(d_model=65, num_heads=8)
    
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        batch_size, seq_len, d_model = 2, 8, 64
        num_heads = 8
        
        attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        attention.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(x)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check attention weights properties (per head)
        for head in range(num_heads):
            head_weights = attention_weights[:, head, :, :]
            assert torch.allclose(head_weights.sum(dim=-1), torch.ones(batch_size, seq_len))
            assert (head_weights >= 0).all()
            assert (head_weights <= 1).all()
    
    def test_forward_pass_with_mask(self):
        """Test forward pass with attention mask."""
        batch_size, seq_len, d_model = 1, 4, 32
        num_heads = 4
        
        attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        attention.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask (1 means attend, 0 means mask out)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()
        
        output, attention_weights = attention(x, mask=mask)
        
        # Check shapes
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
        # Check that mask is applied correctly for all heads
        upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        for head in range(num_heads):
            masked_positions = attention_weights[0, head][upper_triangular]
            assert (masked_positions < 1e-5).all()
    
    def test_attention_weights_storage(self):
        """Test that attention weights are properly stored and retrievable."""
        attention = MultiHeadSelfAttention(d_model=32, num_heads=4)
        x = torch.randn(1, 4, 32)
        
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
        assert stored_weights.shape == (1, 4, 4)
    
    def test_different_head_counts(self):
        """Test attention with different numbers of heads."""
        d_model = 64
        seq_len = 8
        x = torch.randn(1, seq_len, d_model)
        
        for num_heads in [1, 2, 4, 8, 16]:
            attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
            attention.eval()
            
            output, attention_weights = attention(x)
            
            assert output.shape == (1, seq_len, d_model)
            assert attention_weights.shape == (1, num_heads, seq_len, seq_len)
            assert attention.d_head == d_model // num_heads