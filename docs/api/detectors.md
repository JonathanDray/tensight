# Detectors API Reference

Complete API documentation for Tensight detectors.

## GradientDetector

Detect vanishing or exploding gradients.

### Constructor

```python
GradientDetector(threshold_min=1e-6, threshold_max=1e3)
```

**Parameters:**
- `threshold_min` (float): Minimum gradient threshold (default: 1e-6)
- `threshold_max` (float): Maximum gradient threshold (default: 1e3)

### Methods

#### `detect(model, data_loader)`

Detect gradient issues.

**Parameters:**
- `model` (nn.Module): PyTorch model
- `data_loader` (DataLoader): DataLoader for gradient computation

**Returns:**
List of detected issues (empty if none)

**Example:**
```python
from tensight.detectors import GradientDetector

detector = GradientDetector()
issues = detector.detect(model, data_loader)
if issues:
    print("Gradient issues found:", issues)
```

---

## LearningRateDetector

Detect learning rate issues.

### Constructor

```python
LearningRateDetector(threshold_low=1e-6, threshold_high=1.0)
```

**Parameters:**
- `threshold_low` (float): Low learning rate threshold (default: 1e-6)
- `threshold_high` (float): High learning rate threshold (default: 1.0)

### Methods

#### `detect(model, optimizer, data_loader)`

Detect learning rate issues.

**Parameters:**
- `model` (nn.Module): PyTorch model
- `optimizer` (Optimizer): PyTorch optimizer
- `data_loader` (DataLoader): DataLoader

**Returns:**
List of detected issues

**Example:**
```python
from tensight.detectors import LearningRateDetector

detector = LearningRateDetector()
issues = detector.detect(model, optimizer, data_loader)
```

---

## DeadNeuronsDetector

Detect neurons that never activate.

### Constructor

```python
DeadNeuronsDetector(threshold=0.01)
```

**Parameters:**
- `threshold` (float): Activation threshold (default: 0.01)

### Methods

#### `detect(model, data_loader)`

Detect dead neurons.

**Parameters:**
- `model` (nn.Module): PyTorch model
- `data_loader` (DataLoader): DataLoader

**Returns:**
Dictionary with:
- `dead_neurons`: List of dead neuron locations
- `dead_percentage`: Percentage of dead neurons

**Example:**
```python
from tensight.detectors import DeadNeuronsDetector

detector = DeadNeuronsDetector()
results = detector.detect(model, data_loader)
print(f"Dead neurons: {results['dead_percentage']:.2f}%")
```
