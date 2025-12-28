# Watcher

Monitor your model during training.

## Usage

```python
import tensight

watched_model = tensight.watch(model)

# Train normally
for x, y in train_loader:
    output = watched_model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    watched_model.record_loss(loss.item())

# Access monitored data
gradients = watched_model.gradients
activations = watched_model.activations
loss_history = watched_model.loss_history
```

## What it monitors

- **Gradients**: Captured during backward pass
- **Activations**: Captured during forward pass
- **Loss**: Recorded when you call `record_loss()`

## Detectors

Automatically runs detectors for:
- Vanishing/exploding gradients
- Dead neurons
- Activation saturation
- Learning rate issues

