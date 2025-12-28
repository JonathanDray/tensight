# Detectors

Automatic detection of training issues.

## Available detectors

### Gradient detectors
- `detectors.gradient.check_vanishing()`: Detect vanishing gradients
- `detectors.gradient.check_exploding()`: Detect exploding gradients

### Learning rate
- `detectors.learning_rate.check_too_high()`: LR too high
- `detectors.learning_rate.check_too_low()`: LR too low

### Dead neurons
- `detectors.dead_neurons.check()`: Detect dead ReLU neurons

### Initialization
- `detectors.initialization.check()`: Weight initialization issues

### Activation
- `detectors.activation.check_saturation()`: Activation saturation

### Numerical
- `detectors.numerical.check_nan()`: NaN values
- `detectors.numerical.check_inf()`: Inf values

## Usage

```python
from tensight.detectors import run_all_detectors

issues = run_all_detectors(watched_model)
```

