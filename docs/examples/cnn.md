# Example: CNN Analysis

Analyzing a convolutional neural network.

```python
import torch.nn as nn
from tensight.analyzers import ActivationProber

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Probe each layer
prober = ActivationProber(model)
results = prober.probe(
    train_loader, test_loader,
    layer_names=['conv1', 'conv2', 'fc1', 'fc2']
)

# See where class info emerges
for layer, res in results['layer_results'].items():
    print(f"{layer}: {res['test_accuracy']*100:.1f}%")
```
