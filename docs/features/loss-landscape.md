# Loss Landscape Analysis

Visualize and analyze the geometry of the loss function around trained parameters.

## Overview

Loss landscape analysis helps you understand:
- **Sharpness**: How sensitive the loss is to parameter perturbations
- **Generalization**: Flat minima tend to generalize better than sharp minima
- **Training dynamics**: Understand where your model converged

## Basic Usage

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(
    num_points=21,      # Grid resolution (21x21)
    range_val=1.0,      # Perturbation range [-1.0, +1.0]
    use_filter_norm=True,  # Use filter normalization (recommended)
    num_batches=5       # Number of batches for loss computation
)
```

## Understanding Results

The `analyze()` method returns a dictionary with:

- `center_loss`: Loss at the current parameters
- `sharpness`: Average sharpness metric
- `sharpness_max`: Maximum sharpness
- `sharpness_interpretation`: Human-readable interpretation
- `landscape_2d`: 2D array of loss values (for visualization)
- `landscape_min`, `landscape_max`, `landscape_std`: Statistics

## Filter Normalization

Filter normalization (Li et al., 2018) ensures fair comparison across layers with different weight scales. **Always use `use_filter_norm=True`** for meaningful results.

## Interpretation Guide

| Sharpness | Interpretation | Generalization |
|-----------|---------------|----------------|
| < 0.01 | Extremely flat | Excellent |
| 0.01 - 0.1 | Very flat | Good |
| 0.1 - 0.5 | Flat | Good expected |
| 0.5 - 1.0 | Moderate | Okay |
| 1.0 - 5.0 | Sharp | Poor likely |
| > 5.0 | Very sharp | Overfitting likely |

## Example

```python
results = analyzer.analyze(num_points=21, range_val=1.0)

if results['sharpness'] < 0.1:
    print("✅ Model found a flat minimum - good generalization expected!")
else:
    print("⚠️ Model found a sharp minimum - may not generalize well")
```

## Advanced Usage

### Custom Grid Resolution

```python
# Higher resolution for detailed analysis
results = analyzer.analyze(num_points=51, range_val=1.0)
```

### Different Perturbation Range

```python
# Smaller range for local analysis
results = analyzer.analyze(num_points=21, range_val=0.5)
```

## Research Reference

Based on: "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018)

[Read the paper →](https://arxiv.org/abs/1712.09913)
