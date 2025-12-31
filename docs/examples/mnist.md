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

print("Training model...")
for epoch in range(5):
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/5 completed")

# Analyze loss landscape
print("\nAnalyzing loss landscape...")
analyzer = LossLandscapeAnalyzer(model, criterion, train_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)

print(f"Sharpness: {results['sharpness']:.4f}")
print(f"Interpretation: {results['sharpness_interpretation']}")

# Analyze gradient noise
print("\nAnalyzing gradient noise...")
noise_analyzer = GradientNoiseAnalyzer(model, criterion, train_loader)
noise_results = noise_analyzer.analyze(num_batches=30)

print(f"Optimal batch size: {noise_results['optimal_batch_size']}")
print(f"Gradient noise scale: {noise_results['gradient_noise_scale']:.2f}")
```

## Running the Example

See `tensight/tests/test_landscape.py` for a complete working example.

## Expected Output

```
Training model...
Epoch 1/5 completed
Epoch 2/5 completed
Epoch 3/5 completed
Epoch 4/5 completed
Epoch 5/5 completed

Analyzing loss landscape...
🗺️ Analyzing Loss Landscape...
   Grid: 21x21
   Range: [-1.0, +1.0]
   Filter normalization: True
   ...
Sharpness: 0.1350
Interpretation: Flat (good generalization expected)

Analyzing gradient noise...
📊 Analyzing Gradient Noise Scale...
   ...
Optimal batch size: 16
Gradient noise scale: 16.07
```

## Next Steps

- Try different architectures
- Experiment with different hyperparameters
- Compare sharp vs flat models
