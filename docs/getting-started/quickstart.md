# Quick Start

Get started with Tensight in 5 minutes!

## Basic Loss Landscape Analysis

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensight.analyzers import LossLandscapeAnalyzer

# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create dummy data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64)

# Train the model (optional, but recommended)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Analyze loss landscape
analyzer = LossLandscapeAnalyzer(model, criterion, loader)
results = analyzer.analyze(
    num_points=21,      # Grid resolution
    range_val=1.0,      # Perturbation range
    use_filter_norm=True  # Use filter normalization
)

print(f"Sharpness: {results['sharpness']:.4f}")
print(f"Interpretation: {results['sharpness_interpretation']}")
```

## Gradient Noise Analysis

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, criterion, loader)
results = analyzer.analyze(num_batches=30)

print(f"Optimal batch size: {results['optimal_batch_size']}")
print(f"Gradient noise scale: {results['gradient_noise_scale']:.2f}")
```

## Activation Probing

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['layer1', 'layer2', 'layer3']
)

for layer, res in results['layer_results'].items():
    acc = res['test_accuracy'] * 100
    print(f"{layer}: {acc:.1f}%")
```

## Next Steps

- Read the [Loss Landscape guide](../features/loss-landscape.md)
- Check out [examples](../examples/mnist.md)
- Explore the [API reference](../api/analyzers.md)

