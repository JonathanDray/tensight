# API Reference

## Main API

### `tensight.watch(model)`

Wrap a model for monitoring.

**Parameters:**
- `model` (nn.Module): PyTorch model

**Returns:**
- `WatchedModel`: Wrapped model

### `tensight.pre_check(model, data, config)`

Run pre-training checks.

**Parameters:**
- `model` (nn.Module): PyTorch model
- `data`: DataLoader or dataset
- `config` (dict): Training configuration

**Returns:**
- `Report`: Analysis report

## WatchedModel

### Methods

- `forward(x)`: Forward pass
- `record_loss(loss)`: Record loss value

### Attributes

- `gradients`: Dictionary of layer gradients
- `activations`: Dictionary of layer activations
- `loss_history`: List of recorded losses
- `report`: Analysis report

