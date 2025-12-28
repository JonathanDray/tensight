# Loss Landscape

Visualize and analyze the loss landscape around your model's parameters.

## Usage

```python
from tensight.analyzers.loss_landscape import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, loss_fn, data_loader)

results = analyzer.analyze(
    num_points=21,
    range_val=1.0,
    use_filter_norm=True
)

print(f"Sharpness: {results['sharpness']}")
```

## Parameters

- `num_points`: Grid resolution (default: 21)
- `range_val`: Perturbation range (default: 1.0)
- `use_filter_norm`: Filter-wise normalization (recommended)

## Output

- `sharpness`: Average sharpness metric
- `sharpness_max`: Maximum sharpness
- `landscape_2d`: 2D loss surface
- `center_loss`: Loss at current parameters

## Reference

Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018).

