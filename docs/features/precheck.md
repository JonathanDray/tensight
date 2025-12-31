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

**Symptoms:**
- Gradients near zero in early layers
- Model doesn't learn

**Solutions:**
- Use better initialization (Xavier, He)
- Add skip connections
- Use different activation functions

### Exploding Gradients
Gradients become too large, causing training instability.

**Symptoms:**
- Loss becomes NaN
- Gradients explode

**Solutions:**
- Gradient clipping
- Lower learning rate
- Better initialization

### Dead Neurons
Neurons that never activate, reducing model capacity.

**Symptoms:**
- Many neurons with zero activations
- Reduced model capacity

**Solutions:**
- Use Leaky ReLU or other activations
- Better initialization
- Adjust learning rate

### Poor Initialization
Weights initialized in a way that prevents effective learning.

**Solutions:**
- Use PyTorch's default initialization
- Try Xavier or He initialization
- Check weight distributions

### Learning Rate Issues
Learning rate too high (unstable) or too low (slow convergence).

**Solutions:**
- Use learning rate finder
- Start with recommended values
- Use learning rate scheduling

## Example

```python
from tensight import precheck
import torch
import torch.nn as nn

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

## Advanced Usage

### Check Specific Components

```python
# Check only gradients
gradient_issues = precheck.check_gradients(model, data_loader)

# Check only initialization
init_issues = precheck.check_initialization(model)
```

### Custom Thresholds

```python
# Use custom thresholds
issues = precheck.check_model(
    model, 
    sample_input,
    gradient_threshold=1e-6,  # Custom threshold
    activation_threshold=0.01
)
```
