# MNIST Example

Complete example with loss landscape analysis.

## Code

```python
import torch
from torchvision import datasets, transforms
from tensight.analyzers.loss_landscape import LossLandscapeAnalyzer

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

# Define model
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)
)

# Analyze loss landscape
analyzer = LossLandscapeAnalyzer(
    model, 
    torch.nn.CrossEntropyLoss(), 
    train_loader
)

results = analyzer.analyze(
    num_points=21,
    range_val=1.0,
    use_filter_norm=True
)

print(f"Sharpness: {results['sharpness']:.4f}")
```

See `tensight/test_MNIST.py` for full example.

