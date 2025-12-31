# 🔍 Tensight

**See through your PyTorch models** - Advanced analysis and diagnostics toolkit for deep learning models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Tensight provides powerful tools to understand, debug, and optimize your neural networks through advanced analysis techniques including loss landscape visualization, gradient noise analysis, activation probing, and more.

## ✨ Features

### 🗺️ Loss Landscape Analysis
Visualize and analyze the geometry of the loss function around trained parameters. Identify sharp vs flat minima to understand generalization properties.

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0, use_filter_norm=True)
```

### 📊 Gradient Noise Scale
Determine optimal batch sizes by analyzing gradient noise. Based on the theory that optimal batch size scales with gradient noise.

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_batches=30)
print(f"Optimal batch size: {results['optimal_batch_size']}")
```

### 🔬 Activation Probing
Discover where class information is encoded in your network by training linear probes on intermediate activations.

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['layer1', 'layer2', 'layer3']
)
```

### 🛡️ Pre-training Checks
Detect common issues before training: vanishing/exploding gradients, poor initialization, dead neurons, and more.

```python
from tensight import precheck

issues = precheck.check_model(model, sample_input)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JonathanDray/tensight.git
cd tensight

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensight.analyzers import LossLandscapeAnalyzer

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
results = analyzer.analyze(
    num_points=21,
    range_val=1.0,
    use_filter_norm=True
)

print(f"Sharpness: {results['sharpness']:.4f}")
print(f"Interpretation: {results['sharpness_interpretation']}")
```

## 📁 Project Structure

```
tensight/
├── tensight/
│   ├── analyzers/          # Advanced analysis tools
│   │   ├── loss_landscape.py
│   │   ├── gradient_noise.py
│   │   └── activation_probing.py
│   ├── detectors/           # Issue detection
│   │   ├── gradient.py
│   │   ├── learning_rate.py
│   │   └── dead_neurons.py
│   ├── report/             # Report generation
│   │   └── builder.py
│   ├── watcher.py          # Model monitoring
│   └── precheck.py         # Pre-training checks
├── tests/                  # Test suite
│   ├── test_landscape.py
│   ├── test_basic.py
│   └── test_activation_probing.py
├── examples/               # Usage examples
└── README.md
```

## 📚 Examples

### Loss Landscape Visualization

See how your model's loss landscape looks around the trained parameters:

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)

# Results include:
# - landscape_2d: 2D array of loss values
# - sharpness: Average sharpness metric
# - sharpness_max: Maximum sharpness
# - sharpness_interpretation: Human-readable interpretation
```

### Finding Optimal Batch Size

Use gradient noise analysis to find the optimal batch size:

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_batches=30)

print(f"Current batch size: {results['current_batch_size']}")
print(f"Optimal batch size: {results['optimal_batch_size']}")
print(f"Gradient noise scale: {results['gradient_noise_scale']:.2f}")
```

### Understanding Representations

Probe activations to see where class information emerges:

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['conv1', 'conv2', 'fc1', 'fc2']
)

# Check where information is encoded
for layer, res in results['layer_results'].items():
    acc = res['test_accuracy'] * 100
    print(f"{layer}: {acc:.1f}%")
```

## 🧪 Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Set Python path
export PYTHONPATH=/home/jdray/tensight/tensight:$PYTHONPATH

# Run tests
python tensight/tests/test_landscape.py
python tensight/tests/test_basic.py
python tensight/tests/test_activation_probing.py
```

## 📊 Visualizations

Tensight generates beautiful visualizations to help you understand your models:

- **Loss Landscape**: 2D heatmaps and 3D surfaces showing loss geometry
- **Gradient Noise**: Batch size optimization curves
- **Activation Probing**: Layer-wise information encoding analysis

Check out the generated reports:
- `landscape_mnist_report.png`
- `gradient_noise_report.png`
- `probing_report.png`

## 🔬 Research Background

Tensight implements techniques from cutting-edge research:

- **Loss Landscape**: Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018)
- **Gradient Noise**: Implements gradient noise scale theory for optimal batch sizing
- **Activation Probing**: Linear probing to understand representation learning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Jonathan Dray**

- GitHub: [@JonathanDray](https://github.com/JonathanDray)

## 🙏 Acknowledgments

- PyTorch team for the amazing framework
- The research community for the theoretical foundations
- All contributors and users of Tensight

---

**Made with ❤️ for the deep learning community**

