# Activation Probing

Discover where class information is encoded in your network.

## Overview

Activation probing trains linear classifiers on intermediate activations to understand:
- Where class information emerges in the network
- How representations evolve through layers
- Which layers are most informative

## Basic Usage

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['conv1', 'conv2', 'fc1', 'fc2'],
    max_samples=2000
)
```

## Understanding Results

The `probe()` method returns a dictionary with:

- `layer_results`: Dictionary mapping layer names to results
  - `test_accuracy`: Probe accuracy on test set
  - `train_accuracy`: Probe accuracy on train set
- `best_layer`: Layer with highest probe accuracy
- `worst_layer`: Layer with lowest probe accuracy

## Interpretation Guide

| Accuracy | Meaning |
|----------|---------|
| < 30% | Raw features, no class info yet |
| 30-70% | Class information being formed |
| > 70% | Class information fully encoded |

## Example

```python
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['layer1', 'layer2', 'layer3']
)

for layer, res in results['layer_results'].items():
    acc = res['test_accuracy'] * 100
    if acc < 30:
        status = "raw features"
    elif acc < 70:
        status = "forming"
    else:
        status = "encoded"
    print(f"{layer}: {acc:.1f}% ({status})")
```

## Use Cases

### Transfer Learning

Find the best layer to extract features for transfer learning:

```python
results = prober.probe(...)
best_layer = results['best_layer']
print(f"Use {best_layer} for feature extraction")
```

### Architecture Comparison

Compare how different architectures encode information:

```python
# Test different architectures
for arch_name, model in architectures.items():
    prober = ActivationProber(model)
    results = prober.probe(...)
    print(f"{arch_name}: Best layer = {results['best_layer']}")
```

### Debugging

Identify layers that aren't learning useful representations:

```python
results = prober.probe(...)
if results['worst_layer'] in ['layer3', 'layer4']:
    print("⚠️ Middle layers may need attention")
```

## Advanced Usage

### Custom Sample Limits

```python
# Use more samples for better accuracy
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['layer1', 'layer2'],
    max_samples=5000  # More samples
)
```

### Specific Layer Analysis

```python
# Probe only specific layers
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['fc1', 'fc2']  # Only fully connected layers
)
```
