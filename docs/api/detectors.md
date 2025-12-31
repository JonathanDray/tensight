# Detectors API Reference

Complete API documentation for Tensight detectors.

## GradientDetector

Detect vanishing or exploding gradients.

### Constructor

```python
GradientDetector(threshold_min=1e-6, threshold_max=1e3)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_min` | `float` | `1e-6` | Minimum gradient threshold |
| `threshold_max` | `float` | `1e3` | Maximum gradient threshold |

### Methods

#### `detect(model, data_loader)`

Detect gradient issues.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `data_loader` | `DataLoader` | DataLoader for gradient computation |

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_low` | `float` | `1e-6` | Low learning rate threshold |
| `threshold_high` | `float` | `1.0` | High learning rate threshold |

### Methods

#### `detect(model, optimizer, data_loader)`

Detect learning rate issues.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `optimizer` | `Optimizer` | PyTorch optimizer |
| `data_loader` | `DataLoader` | DataLoader |

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.01` | Activation threshold |

### Methods

#### `detect(model, data_loader)`

Detect dead neurons.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `data_loader` | `DataLoader` | DataLoader |

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
