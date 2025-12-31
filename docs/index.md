# 🔍 Tensight

**See through your PyTorch models** - Advanced analysis toolkit for deep learning.

## ✨ Features

- 🗺️ [Loss Landscape Analysis](features/loss-landscape/) - Visualize loss geometry, identify sharp vs flat minima
- 📊 [Gradient Noise Scale](features/gradient-noise/) - Find optimal batch size automatically
- 🔬 [Activation Probing](features/activation-probing/) - Discover where information is encoded
- 🛡️ [Pre-training Checks](features/precheck/) - Detect issues before training

## 🚀 Quick Start

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze()
print(f"Sharpness: {results['sharpness']:.4f}")
```

## 📚 Documentation

- [Installation](getting-started/installation/)
- [Quick Start](getting-started/quickstart/)
- [API Reference](api/analyzers/)
- [Examples](examples/mnist/)

## 📝 License

MIT License
