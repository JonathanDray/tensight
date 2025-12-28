# CNN Example

Example with convolutional neural network.

```python
import torch
import torch.nn as nn
import tensight

# Define CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN()
watched_model = tensight.watch(model)

# Pre-check
report = tensight.pre_check(model, train_loader, config={'lr': 0.001})
report.display()

# Train
for x, y in train_loader:
    output = watched_model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    optimizer.step()
```

