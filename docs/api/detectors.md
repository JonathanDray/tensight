# API Reference - Detectors

## PreTrainingChecker

Detect common issues before training starts.

```python
from tensight.detectors import PreTrainingChecker

checker = PreTrainingChecker(model)
results = checker.run_all_checks(sample_input)
```

**Checks performed:**
- Vanishing/exploding gradients
- Dead ReLU neurons
- Poor weight initialization
- NaN/Inf detection
