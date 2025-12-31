# Pre-training Checks

Detect common issues before training starts.

## Why?

Catch problems early. Save hours of wasted training time.

## Usage

```python
from tensight.detectors import PreTrainingChecker

checker = PreTrainingChecker(model)
results = checker.run_all_checks(sample_input)
```

## Checks Performed

- **Gradient Flow**: Detect vanishing/exploding gradients
- **Dead Neurons**: Find ReLU neurons that never activate
- **Weight Init**: Check for poor initialization
- **NaN/Inf**: Detect numerical instabilities

## Output

Each check returns status (✅/⚠️/❌) and actionable suggestions.
