# Pre-check

Analyze model, data, and configuration before training starts.

## Usage

```python
import tensight

report = tensight.pre_check(
    model=model,
    data=train_loader,
    config={'lr': 0.01, 'batch_size': 32}
)

report.display()
```

## Checks performed

- **Model**: Parameter count, trainable params, architecture
- **Data**: Sample size, distribution, quality issues
- **Config**: Learning rate, batch size, optimizer settings
- **Hardware**: GPU availability, memory estimation
- **Test**: Forward/backward pass validation

## Output

Pre-check generates a report with:
- Statistics
- Warnings
- Recommendations

