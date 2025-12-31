# Gradient Noise Scale

Determine optimal batch sizes by analyzing gradient noise.

## Overview

Gradient noise analysis helps you:
- Find the optimal batch size for your model
- Understand training efficiency
- Balance between gradient variance and compute cost

## Theory

The gradient noise scale measures the variance in gradients across batches. Optimal batch size scales with this noise - more noise means you can use larger batches.

!!! tip "Why it matters"
    Using the optimal batch size can significantly improve training efficiency and model performance.

## Basic Usage

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_batches=30)

print(f"Current batch size: {results['current_batch_size']}")
print(f"Optimal batch size: {results['optimal_batch_size']}")
print(f"Gradient noise scale: {results['gradient_noise_scale']:.2f}")
```

## Understanding Results

| Key | Description |
|-----|-------------|
| `gradient_noise_scale` | Measure of gradient variance |
| `optimal_batch_size` | Recommended batch size |
| `current_batch_size` | Your current batch size |
| `efficiency` | How close you are to optimal (100% = optimal) |
| `mean_gradient_norm` | Average gradient norm |
| `gradient_variance` | Gradient variance |

## Example

```python
results = analyzer.analyze(num_batches=30)

if results['optimal_batch_size'] < results['current_batch_size']:
    print(f"💡 Try reducing batch size to {results['optimal_batch_size']}")
elif results['optimal_batch_size'] > results['current_batch_size']:
    print(f"💡 Try increasing batch size to {results['optimal_batch_size']}")
else:
    print("✅ Batch size is optimal!")
```

!!! success "Expected Output"
    ```
    📊 Analyzing Gradient Noise Scale...
       Collecting gradients: 1/30
       ...
    
    Gradient Noise Scale: 16.07
    Optimal batch size: 16
    Efficiency: 100.0%
    ```

## Tips

- Use more batches (`num_batches=30-50`) for more accurate estimates
- Re-run analysis after significant training progress
- Consider compute constraints when choosing batch size

## Advanced Usage

### More Accurate Estimates

```python
# Use more batches for better accuracy
results = analyzer.analyze(num_batches=50)
```

### Batch Size Comparison

```python
# Test different batch sizes
for batch_size in [16, 32, 64, 128]:
    loader = DataLoader(dataset, batch_size=batch_size)
    analyzer = GradientNoiseAnalyzer(model, criterion, loader)
    results = analyzer.analyze()
    print(f"Batch {batch_size}: Optimal = {results['optimal_batch_size']}")
```
