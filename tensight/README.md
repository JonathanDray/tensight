# 🔍 Tensight

**Deep Learning Debugger - See through your models.**

Automatic detection and diagnosis of ML training problems with specific, actionable suggestions.

## Features

- 🔬 **Pre-Check**: Analyze model, data, and config *before* training
- 👁️ **Watch**: Monitor gradients, activations, and loss during training
- 📊 **Diagnose**: Get actionable suggestions to fix problems
- 🗺️ **Loss Landscape**: Visualize the geometry of your loss function
- 📈 **Gradient Noise**: Find optimal batch size

## Installation

```bash
pip install tensight
```

Or from source:

```bash
git clone https://github.com/yourusername/tensight.git
cd tensight
pip install -e .
```

## Quick Start

### Pre-Check (Before Training)

```python
import tensight

# Analyze before spending compute
report = tensight.pre_check(
    model=model,
    data=(X_train, y_train),
    config={'lr': 0.01, 'batch_size': 32, 'epochs': 100}
)

if not report.can_train:
    print("Fix these issues first!")
```

### Watch (During Training)

```python
import tensight

# Wrap your model
model = tensight.watch(model)

# Train normally
for epoch in range(epochs):
    loss = train_step(model, data)
    model.record_loss(loss)

# Get diagnosis
report = model.diagnose()
```

### Loss Landscape Analysis

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, loss_fn, train_loader)
results = analyzer.analyze(num_points=21, use_filter_norm=True)

print(f"Sharpness: {results['sharpness']}")
# Sharp minimum = poor generalization
# Flat minimum = good generalization
```

## What Gets Detected

| Problem | Severity | Suggestion |
|---------|----------|------------|
| Vanishing Gradients | ⚠️ Warning | Use ReLU/GELU, add BatchNorm |
| Exploding Gradients | 🔴 Error | Use gradient clipping |
| Dead Neurons | ⚠️ Warning | Use Leaky ReLU |
| NaN Values | 🔴 Error | Reduce learning rate |
| Loss Explosion | 🔴 Error | Reduce learning rate by 10x |
| Loss Stagnation | ⚠️ Warning | Increase LR, try Adam |
| Class Imbalance | ⚠️ Warning | Use class_weight |

## Advanced Analysis

### Loss Landscape (Li et al., NeurIPS 2018)

Visualize the shape of your loss function:
- **Sharp minima** → Poor generalization
- **Flat minima** → Good generalization

### Gradient Noise Scale (OpenAI, 2018)

Find optimal batch size:
- Batch size < B_noise → Inefficient
- Batch size ≈ B_noise → Optimal

## Example Output

```
╔══════════════════════════════════════════════════════════╗
║           🔍 TENSIGHT DIAGNOSTIC REPORT                  ║
╚══════════════════════════════════════════════════════════╝

🏥 Health Score: 🟡 RISKY
📣 Recommendation: Fix warnings first

⚠️ Problems Detected: 2

🟡 Vanishing Gradient
   Layer 'fc2': gradients are very small (mean=1.23e-08)
   💡 Use ReLU/GELU instead of Sigmoid/Tanh, add BatchNorm
   📄 Ref: Glorot & Bengio, 2010

🟡 Learning Rate High
   lr=0.01 may cause instability
   💡 Try lr=0.001 if training is unstable
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy

## Research References

- Li et al., "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018
- McCandlish et al., "An Empirical Model of Large-Batch Training", OpenAI 2018
- Glorot & Bengio, "Understanding difficulty of training deep feedforward neural networks", 2010
- Pascanu et al., "On the difficulty of training Recurrent Neural Networks", 2013

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

---

🔍 **Tensight** - See through your models.
