"""Test script to verify SelfAttention implementation."""

import torch
import sys
import os

# Add the src directory to the path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from attentions import VanillaSelfAttention, scaled_dot_product_attention


def test_scaled_dot_product_attention():
    """Test the core scaled dot-product attention function."""
    print("Testing scaled_dot_product_attention...")
    
    # Create test tensors
    batch_size, seq_len, d_model = 2, 4, 8
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Test basic functionality
    output, attention_weights = scaled_dot_product_attention(query, key, value)
    
    # Verify shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), f"Expected attention weights shape {(batch_size, seq_len, seq_len)}, got {attention_weights.shape}"
    
    # Verify attention weights sum to 1 along the last dimension
    weights_sum = attention_weights.sum(dim=-1)
    expected_sum = torch.ones(batch_size, seq_len)
    assert torch.allclose(weights_sum, expected_sum, atol=1e-6), "Attention weights don't sum to 1"
    
    print("✓ scaled_dot_product_attention test passed")


def test_vanilla_self_attention():
    """Test the VanillaSelfAttention implementation."""
    print("Testing VanillaSelfAttention...")
    
    # Create test configuration
    d_model = 64
    batch_size, seq_len = 2, 8
    
    # Initialize the attention module
    attention = VanillaSelfAttention(d_model=d_model, dropout=0.1, temperature=1.0)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    output, attention_weights = attention.forward(x, x, x)
    
    # Verify shapes
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert attention_weights.shape == (batch_size, seq_len, seq_len), f"Expected attention weights shape {(batch_size, seq_len, seq_len)}, got {attention_weights.shape}"
    
    # Test attention weights retrieval
    retrieved_weights = attention.get_attention_weights()
    assert torch.equal(retrieved_weights, attention_weights.detach()), "Retrieved attention weights don't match"
    
    # Test with self-attention convenience method
    output_self, attention_weights_self = attention.forward_self(x)
    assert output_self.shape == output.shape, "Self-attention output shape mismatch"
    
    # Test configuration retrieval
    config = attention.get_config()
    expected_keys = {"d_model", "dropout", "bias", "temperature"}
    assert set(config.keys()) == expected_keys, f"Config keys mismatch. Expected {expected_keys}, got {set(config.keys())}"
    assert config["d_model"] == d_model, f"d_model mismatch in config"
    
    print("✓ VanillaSelfAttention test passed")


def test_attention_with_mask():
    """Test attention with masking."""
    print("Testing attention with masking...")
    
    d_model = 32
    batch_size, seq_len = 1, 4
    
    attention = VanillaSelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create a causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Test with mask
    output, attention_weights = attention.forward(x, x, x, mask=mask)
    
    # Verify that attention weights are zero where mask is False
    # Note: we need to check the upper triangular part is close to zero
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attention_weights[0, i, j].item() < 1e-6, f"Attention weight at ({i}, {j}) should be near zero with causal mask"
    
    print("✓ Attention masking test passed")


def test_gradient_flow():
    """Test that gradients flow properly through the attention mechanism."""
    print("Testing gradient flow...")
    
    d_model = 16
    batch_size, seq_len = 1, 3
    
    attention = VanillaSelfAttention(d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Forward pass
    output, _ = attention.forward(x, x, x)
    
    # Compute a simple loss (sum of outputs)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "Input gradients should exist"
    assert attention.w_q.weight.grad is not None, "Query weight gradients should exist"
    assert attention.w_k.weight.grad is not None, "Key weight gradients should exist"
    assert attention.w_v.weight.grad is not None, "Value weight gradients should exist"
    assert attention.w_o.weight.grad is not None, "Output weight gradients should exist"
    
    print("✓ Gradient flow test passed")


def main():
    """Run all tests."""
    print("Running SelfAttention implementation tests...\n")
    
    try:
        test_scaled_dot_product_attention()
        test_vanilla_self_attention()
        test_attention_with_mask()
        test_gradient_flow()
        
        print("\n🎉 All tests passed! SelfAttention implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)