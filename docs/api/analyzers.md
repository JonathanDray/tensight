# Analyzers API Reference

Complete API documentation for Tensight analyzers.

## LossLandscapeAnalyzer

Analyze the loss landscape around trained parameters.

### Constructor

```python
LossLandscapeAnalyzer(model, loss_fn, data_loader, device=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Trained PyTorch model |
| `loss_fn` | `Callable` | Loss function (e.g., `nn.CrossEntropyLoss()`) |
| `data_loader` | `DataLoader` | DataLoader for computing loss |
| `device` | `torch.device`, optional | Device to use (auto-detected if None) |

### Methods

#### `analyze(num_points=21, range_val=1.0, use_filter_norm=True, num_batches=5, seed=42, debug=False)`

Compute loss landscape and sharpness metrics.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_points` | `int` | `21` | Grid resolution |
| `range_val` | `float` | `1.0` | Perturbation range |
| `use_filter_norm` | `bool` | `True` | Use filter normalization |
| `num_batches` | `int` | `5` | Number of batches for loss computation |
| `seed` | `int` | `42` | Random seed |
| `debug` | `bool` | `False` | Enable debug output |

**Returns:**

Dictionary with:
- `center_loss`: Loss at current parameters
- `sharpness`: Average sharpness
- `sharpness_max`: Maximum sharpness
- `sharpness_interpretation`: Human-readable interpretation
- `landscape_2d`: 2D array of loss values
- `landscape_min`, `landscape_max`, `landscape_std`: Statistics

**Example:**

```python
analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_points=21, range_val=1.0)
print(results['sharpness'])
```

---

## GradientNoiseAnalyzer

Analyze gradient noise to determine optimal batch size.

### Constructor

```python
GradientNoiseAnalyzer(model, loss_fn, data_loader, device=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `loss_fn` | `Callable` | Loss function |
| `data_loader` | `DataLoader` | DataLoader for gradient computation |
| `device` | `torch.device`, optional | Device to use |

### Methods

#### `analyze(num_batches=30)`

Analyze gradient noise scale.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_batches` | `int` | `30` | Number of batches to analyze |

**Returns:**

Dictionary with:
- `gradient_noise_scale`: Measure of gradient variance
- `optimal_batch_size`: Recommended batch size
- `current_batch_size`: Current batch size
- `efficiency`: Efficiency percentage (100% = optimal)
- `mean_gradient_norm`: Average gradient norm
- `gradient_variance`: Gradient variance

**Example:**

```python
analyzer = GradientNoiseAnalyzer(model, criterion, data_loader)
results = analyzer.analyze(num_batches=30)
print(f"Optimal batch size: {results['optimal_batch_size']}")
```

---

## ActivationProber

Probe activations to understand representation learning.

### Constructor

```python
ActivationProber(model, device=None)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `device` | `torch.device`, optional | Device to use |

### Methods

#### `probe(train_loader, test_loader, layer_names, max_samples=2000)`

Train linear probes on activations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_loader` | `DataLoader` | - | Training data |
| `test_loader` | `DataLoader` | - | Test data |
| `layer_names` | `List[str]` | - | Names of layers to probe |
| `max_samples` | `int` | `2000` | Maximum samples per class |

**Returns:**

Dictionary with:
- `layer_results`: Results for each layer
  - `test_accuracy`: Probe accuracy on test set
  - `train_accuracy`: Probe accuracy on train set
- `best_layer`: Layer with highest accuracy
- `worst_layer`: Layer with lowest accuracy

**Example:**

```python
prober = ActivationProber(model)
results = prober.probe(
    train_loader=train_loader,
    test_loader=test_loader,
    layer_names=['layer1', 'layer2', 'layer3']
)
print(results['best_layer'])
```
