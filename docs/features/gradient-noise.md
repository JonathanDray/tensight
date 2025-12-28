# Gradient Noise Scale

Compute optimal batch size using gradient noise scale.

## Usage

```python
from tensight.analyzers.gradient_noise import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, loss_fn, data_loader)

results = analyzer.analyze(num_batches=10)

print(f"Noise scale: {results['noise_scale']}")
print(f"Optimal batch size: {results['optimal_batch_size']}")
```

## Theory

Gradient Noise Scale = tr(Σ) / ||G||²

Where:
- Σ = covariance of gradients
- G = mean gradient

## Output

- `noise_scale`: Gradient noise scale value
- `optimal_batch_size`: Recommended batch size
- `current_batch_size`: Current batch size
- `recommendation`: Batch size suggestion

## Reference

Based on "An Empirical Model of Large-Batch Training" (OpenAI).

