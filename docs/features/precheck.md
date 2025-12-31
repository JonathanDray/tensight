# Pre-training Checks

Detect common issues before training starts.

## Overview

Pre-training checks help you catch problems early:
- Vanishing/exploding gradients
- Poor weight initialization
- Dead neurons
- Learning rate issues
- Numerical instabilities

## Basic Usage

```python
from tensight import precheck

# Check model before training
sample_input = torch.randn(1, 784)  # Example input
issues = precheck.check_model(model, sample_input)

if issues:
    print("⚠️ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Model looks good!")
```

## Common Issues Detected

### Vanishing Gradients
Gradients become too small, preventing learning in early layers.

### Exploding Gradients
Gradients become too large, causing training instability.

### Dead Neurons
Neurons that never activate, reducing model capacity.

### Poor Initialization
Weights initialized in a way that prevents effective learning.

### Learning Rate Issues
Learning rate too high (unstable) or too low (slow convergence).

## Example

```python
from tensight import precheck
import torch

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

sample_input = torch.randn(1, 784)
issues = precheck.check_model(model, sample_input)

if not issues:
    print("✅ Ready to train!")
else:
    print("⚠️ Fix issues before training:")
    for issue in issues:
        print(f"  - {issue}")
```

