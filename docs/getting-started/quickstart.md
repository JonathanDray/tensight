# Quick Start

Get started with Tensight in 2 minutes.

## 1. Import

```python
from tensight.analyzers import LossLandscapeAnalyzer, GradientNoiseAnalyzer
```

## 2. Analyze Loss Landscape

```python
analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze()
print(f"Sharpness: {results['sharpness']:.4f}")
```

## 3. Find Optimal Batch Size

```python
analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze()
print(f"Optimal batch: {results['optimal_batch_size']}")
```

## 4. Probe Activations

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(train_loader, test_loader)
```

## Next Steps

- [Loss Landscape](../features/loss-landscape/) - Detailed guide
- [API Reference](../api/analyzers/) - Full API docs
- [Examples](../examples/mnist/) - Complete examples
