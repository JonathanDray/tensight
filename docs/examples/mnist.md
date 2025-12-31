# MNIST Example

Complete example using Tensight on MNIST dataset.

## Full Code

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensight.analyzers import LossLandscapeAnalyzer, GradientNoiseAnalyzer

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Create model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# Analyze loss landscape
analyzer = LossLandscapeAnalyzer(model, criterion, train_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)

print(f"Sharpness: {results['sharpness']:.4f}")
print(f"Interpretation: {results['sharpness_interpretation']}")

# Analyze gradient noise
noise_analyzer = GradientNoiseAnalyzer(model, criterion, train_loader)
noise_results = noise_analyzer.analyze(num_batches=30)

print(f"Optimal batch size: {noise_results['optimal_batch_size']}")
```

## Running the Example

See `tensight/tests/test_landscape.py` for a complete working example.

