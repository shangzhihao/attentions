# Attentions 🔍

A modern, extensible PyTorch library for attention mechanisms in transformer models and deep learning architectures. Designed for both research purposes with clean, well-documented, and modular code.

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
from attentions import VanillaSelfAttention

# Initialize attention module
d_model = 64
attention = VanillaSelfAttention(
    d_model=d_model,
    dropout=0.1,
    temperature=1.0
)

# Create input tensor
batch_size, seq_len = 2, 10
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output, attention_weights = attention.forward(x, x, x)
print(f"Output shape: {output.shape}")  # [2, 10, 64]
print(f"Attention weights shape: {attention_weights.shape}")  # [2, 10, 10]

# Using self-attention convenience method
output, weights = attention.forward_self(x)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=attentions --cov-report=html

# Run specific test file
python test_attention.py
```
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
