# 🔍 Tensight

**See through your PyTorch models** - Advanced analysis and diagnostics toolkit for deep learning models.

Tensight provides powerful tools to understand, debug, and optimize your neural networks through advanced analysis techniques.

## ✨ Key Features

### 🗺️ [Loss Landscape Analysis](features/loss-landscape.md)
Visualize and analyze the geometry of the loss function around trained parameters. Identify sharp vs flat minima to understand generalization properties.

### 📊 [Gradient Noise Scale](features/gradient-noise.md)
Determine optimal batch sizes by analyzing gradient noise. Based on the theory that optimal batch size scales with gradient noise.

### 🔬 [Activation Probing](features/activation-probing.md)
Discover where class information is encoded in your network by training linear probes on intermediate activations.

### 🛡️ [Pre-training Checks](features/precheck.md)
Detect common issues before training: vanishing/exploding gradients, poor initialization, dead neurons, and more.

## 🚀 Quick Start

```python
from tensight.analyzers import LossLandscapeAnalyzer
import torch.nn as nn
from torch.utils.data import DataLoader

# Your model and data
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
criterion = nn.CrossEntropyLoss()
data_loader = DataLoader(dataset, batch_size=64)

# Analyze loss landscape
analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)

print(f"Sharpness: {results['sharpness']:.4f}")
```

👉 **[Get Started →](getting-started/installation.md)**

## 📚 Documentation

- **[Installation Guide](getting-started/installation.md)** - Get started with Tensight
- **[Quick Start Tutorial](getting-started/quickstart.md)** - Your first analysis
- **[Features Documentation](features/loss-landscape.md)** - Detailed feature guides
- **[API Reference](api/analyzers.md)** - Complete API documentation
- **[Examples](examples/mnist.md)** - Real-world examples

## 🔬 Research Background

Tensight implements techniques from cutting-edge research:

- **Loss Landscape**: Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018)
- **Gradient Noise**: Implements gradient noise scale theory for optimal batch sizing
- **Activation Probing**: Linear probing to understand representation learning

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details.

## 📝 License

This project is licensed under the MIT License.

---

**Made with ❤️ for the deep learning community**
