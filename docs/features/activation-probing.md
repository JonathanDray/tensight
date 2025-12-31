# Activation Probing

Discover where information is encoded in your network.

## Why?

Understand which layers encode what. Useful for transfer learning and debugging.

## Usage

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(
    train_loader, test_loader,
    layer_names=['layer1', 'layer2', 'layer3']
)

for layer, res in results['layer_results'].items():
    print(f"{layer}: {res['test_accuracy']*100:.1f}%")
```

## Interpretation

- Low accuracy → raw features, no class info yet
- Rising accuracy → class info being formed
- High accuracy → class fully encoded

## Reference

Based on "Understanding intermediate layers using linear classifier probes" (Alain & Bengio, 2016)
