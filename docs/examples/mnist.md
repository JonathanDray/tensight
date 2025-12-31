# Example: MNIST Analysis

Complete example analyzing a model trained on MNIST.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensight.analyzers import LossLandscapeAnalyzer, ActivationProber

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Simple model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# Train your model first...

# Analyze loss landscape
analyzer = LossLandscapeAnalyzer(model, nn.CrossEntropyLoss(), train_loader)
results = analyzer.analyze()

# Probe activations
prober = ActivationProber(model)
probe_results = prober.probe(train_loader, test_loader)
```
