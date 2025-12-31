# API Reference - Analyzers

## LossLandscapeAnalyzer

Visualize the loss landscape around trained parameters.

```python
from tensight.analyzers import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, loss_fn, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)
```

**Returns:** `sharpness`, `landscape_2d`, `center_loss`

## GradientNoiseAnalyzer

Find optimal batch size by analyzing gradient noise.

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, loss_fn, data_loader)
results = analyzer.analyze(num_batches=20)
```

**Returns:** `gradient_noise_scale`, `optimal_batch_size`, `efficiency`

## ActivationProber

Probe layers to discover where information is encoded.

```python
from tensight.analyzers import ActivationProber

prober = ActivationProber(model)
results = prober.probe(train_loader, test_loader, layer_names=['fc1', 'fc2'])
```

**Returns:** `layer_results` with accuracy per layer
