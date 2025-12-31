# Analyzers API Reference

## LossLandscapeAnalyzer

Analyze the loss landscape around trained parameters.

### `__init__(model, loss_fn, data_loader, device=None)`

Initialize the analyzer.

**Parameters:**
- `model` (nn.Module): Trained PyTorch model
- `loss_fn` (Callable): Loss function (e.g., `nn.CrossEntropyLoss()`)
- `data_loader` (DataLoader): DataLoader for computing loss
- `device` (torch.device, optional): Device to use (auto-detected if None)

### `analyze(num_points=21, range_val=1.0, use_filter_norm=True, num_batches=5, seed=42, debug=False)`

Compute loss landscape and sharpness metrics.

**Parameters:**
- `num_points` (int): Grid resolution (default: 21)
- `range_val` (float): Perturbation range (default: 1.0)
- `use_filter_norm` (bool): Use filter normalization (default: True)
- `num_batches` (int): Number of batches for loss computation (default: 5)
- `seed` (int): Random seed (default: 42)
- `debug` (bool): Enable debug output (default: False)

**Returns:**
Dictionary with:
- `center_loss`: Loss at current parameters
- `sharpness`: Average sharpness
- `sharpness_max`: Maximum sharpness
- `sharpness_interpretation`: Human-readable interpretation
- `landscape_2d`: 2D array of loss values
- `landscape_min`, `landscape_max`, `landscape_std`: Statistics

## GradientNoiseAnalyzer

Analyze gradient noise to determine optimal batch size.

### `__init__(model, loss_fn, data_loader, device=None)`

Initialize the analyzer.

### `analyze(num_batches=30)`

Analyze gradient noise scale.

**Parameters:**
- `num_batches` (int): Number of batches to analyze (default: 30)

**Returns:**
Dictionary with:
- `gradient_noise_scale`: Measure of gradient variance
- `optimal_batch_size`: Recommended batch size
- `current_batch_size`: Current batch size
- `efficiency`: Efficiency percentage (100% = optimal)

## ActivationProber

Probe activations to understand representation learning.

### `__init__(model, device=None)`

Initialize the prober.

### `probe(train_loader, test_loader, layer_names, max_samples=2000)`

Train linear probes on activations.

**Parameters:**
- `train_loader` (DataLoader): Training data
- `test_loader` (DataLoader): Test data
- `layer_names` (List[str]): Names of layers to probe
- `max_samples` (int): Maximum samples per class (default: 2000)

**Returns:**
Dictionary with:
- `layer_results`: Results for each layer
- `best_layer`: Layer with highest accuracy
- `worst_layer`: Layer with lowest accuracy

