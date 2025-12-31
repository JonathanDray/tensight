# Gradient Noise Scale

Find the optimal batch size for your model automatically.

## Why?

Too small batch → wasting compute. Too large batch → diminishing returns.

## Usage

```python
from tensight.analyzers import GradientNoiseAnalyzer

analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_batches=20)

print(f"Current batch: {results['current_batch_size']}")
print(f"Optimal batch: {results['optimal_batch_size']}")
print(f"Efficiency: {results['efficiency']*100:.0f}%")
```

## Output

- `gradient_noise_scale`: The B_noise value
- `optimal_batch_size`: Recommended batch size
- `efficiency`: How efficient your current batch size is (0-1)

## Reference

Based on "An Empirical Model of Large-Batch Training" (McCandlish et al., OpenAI 2018)
