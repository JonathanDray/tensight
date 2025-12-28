# Quickstart

## Basic usage

```python
import torch
import torch.nn as nn
import tensight

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Wrap with tensight
watched_model = tensight.watch(model)

# Run pre-checks
config = {'lr': 0.01, 'batch_size': 32}
report = tensight.pre_check(model, train_loader, config)

# Train as usual
optimizer = torch.optim.Adam(model.parameters())
for x, y in train_loader:
    output = watched_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    watched_model.record_loss(loss.item())
```

