# Tensight

**See through your models.**

Analysis and diagnostics tool for PyTorch models.

## Quick Start

```python
import tensight

# Wrap your model
model = tensight.watch(model)

# Run pre-training checks
tensight.pre_check(model, data, config)
```

## Features

- **Pre-check**: Analyze model, data, and config before training
- **Watcher**: Monitor gradients, activations, and loss during training
- **Detectors**: Automatic detection of common training issues
- **Analyzers**: Advanced analysis tools for deep learning research

## Installation

```bash
pip install -e .
```

