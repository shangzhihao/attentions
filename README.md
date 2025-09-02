# Attentions 🔍

A modern, extensible PyTorch library for attention mechanisms in transformer models and deep learning architectures. Designed for educational and research purposes with clean, well-documented, and modular code.

## 📋 Attention Mechanisms

### ✅ Currently Implemented

| Mechanism | Class | Description | Use Case |
|-----------|-------|-------------|----------|
| **Vanilla Self-Attention** | `VanillaSelfAttention` | Standard scaled dot-product self-attention | Basic transformer building block |
| **Multi-Head Self-Attention** | `MultiHeadSelfAttention` | Parallel attention heads with different representations | Standard transformer layers |
| **Local Self-Attention** | `LocalSelfAttention` | Windowed attention with configurable window size | Long sequences, O(n×w) complexity |
| **Grouped Self-Attention** | `GroupedSelfAttention` | Memory-efficient attention with shared K,V heads | Efficient transformers (GQA/MQA) |
| **Dilated Self-Attention** | `DilatedSelfAttention` | Sparse attention with dilation patterns | Structured sequences, long-range deps |
| **Linear Self-Attention** | `LinearSelfAttention` | Linear complexity attention using kernel methods | O(n) complexity for very long sequences |
| **Block Self-Attention** | `BlockSelfAttention` | Block-wise sparse attention patterns | Hierarchical attention, document modeling |
| **ALiBi Self-Attention** | `ALiBiSelfAttention` | Attention with linear bias for positions | Length extrapolation capabilities |

### 🚧 Planned Implementations

| Mechanism | Description | Benefits |
|-----------|-------------|----------|
| **Rotary Position** | Attention with rotary positional embeddings | Better positional understanding |

## 🚀 Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/yourusername/attentions.git
cd attentions
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from attentions import (
    VanillaSelfAttention,
    MultiHeadSelfAttention,
    LocalSelfAttention,
    GroupedSelfAttention,
    DilatedSelfAttention
)

# Initialize different attention mechanisms
d_model = 128
seq_len = 512
batch_size = 4

# Vanilla Self-Attention
vanilla_attn = VanillaSelfAttention(d_model=d_model)

# Multi-Head Self-Attention
multi_head_attn = MultiHeadSelfAttention(
    d_model=d_model,
    num_heads=8
)

# Local Self-Attention (for long sequences)
local_attn = LocalSelfAttention(
    d_model=d_model,
    window_size=64,
    num_heads=8
)

# Grouped Self-Attention (memory efficient)
grouped_attn = GroupedSelfAttention(
    d_model=d_model,
    num_query_heads=8,
    num_kv_heads=2  # Shared K,V heads for efficiency
)

# Dilated Self-Attention (sparse patterns)
dilated_attn = DilatedSelfAttention(
    d_model=d_model,
    dilation_rate=4,
    num_heads=8
)

# Create input tensor
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass (same API for all mechanisms)
output, attention_weights = vanilla_attn(x)
print(f"Output shape: {output.shape}")  # [4, 512, 128]
print(f"Attention weights shape: {attention_weights.shape}")  # [4, 1, 512, 512]

# Multi-head attention
output, weights = multi_head_attn(x)
print(f"Multi-head weights shape: {weights.shape}")  # [4, 8, 512, 512]

# Using attention masks
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
masked_output, masked_weights = multi_head_attn(x, mask=causal_mask)
```

## 📊 Performance Comparison

| Mechanism | Time Complexity | Memory Complexity | Best Use Case |
|-----------|----------------|-------------------|---------------|
| Vanilla | O(n²) | O(n²) | Short sequences (< 512) |
| Multi-Head | O(n²) | O(n²) | Standard transformer layers |
| Local | O(n×w) | O(n×w) | Long sequences with local patterns |
| Grouped | O(n²) | O(n²/g) | Memory-constrained scenarios |
| Dilated | O(n×d) | O(n×d) | Structured/periodic patterns |
| Linear | O(n) | O(n) | Very long sequences (> 4K tokens) |
| Block | O(b×(n/b)²) | O(b×(n/b)²) | Memory-efficient long sequences |
| ALiBi | O(n²) | O(n²) | Length extrapolation tasks |

*Where n=sequence length, w=window size, g=group ratio, d=dilation connections, b=number of blocks*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
