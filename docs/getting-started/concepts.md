# Core Concepts

## Watcher

The `watch()` function wraps your model to monitor:
- Gradients (vanishing/exploding)
- Activations (dead neurons, saturation)
- Loss history

## Pre-check

Analyzes before training:
- Model architecture (parameter count, ratios)
- Data quality (NaN, outliers, distribution)
- Configuration (learning rate, batch size)
- Hardware compatibility

## Detectors

Automatic detection of:
- Gradient issues
- Learning rate problems
- Dead neurons
- Initialization issues
- Numerical instability

## Analyzers

Advanced research tools:
- Loss landscape visualization
- Gradient noise scale
- Hessian eigenspectrum
- Fisher information matrix
- And more...

