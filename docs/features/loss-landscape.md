# Loss Landscape Analysis

Visualize the geometry of the loss function around trained parameters.

## Why?

Sharp minima → poor generalization. Flat minima → better generalization.

## Usage

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)

print(f"Sharpness: {results['sharpness']:.4f}")
print(f"Interpretation: {results['sharpness_interpretation']}")
```

## Output

- `landscape_2d`: 2D numpy array for visualization
- `sharpness`: Average sharpness metric
- `sharpness_max`: Maximum sharpness

## Reference

Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018)
