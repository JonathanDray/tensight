# 🔍 Tensight

<div style="text-align: center; margin: 2rem 0;">
  <h2 style="color: #7c3aed; font-size: 2.5rem; margin-bottom: 1rem;">
    See through your PyTorch models
  </h2>
  <p style="font-size: 1.2rem; color: #6b7280; max-width: 800px; margin: 0 auto;">
    Advanced analysis and diagnostics toolkit for deep learning models. 
    Understand, debug, and optimize your neural networks with cutting-edge techniques.
  </p>
</div>

---

## ✨ Key Features

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __Loss Landscape Analysis__

    ---

    Visualize and analyze the geometry of the loss function around trained parameters. 
    Identify sharp vs flat minima to understand generalization properties.

    [:octicons-arrow-right-24: Learn more](features/loss-landscape.md)

-   :material-chart-scatter-plot:{ .lg .middle } __Gradient Noise Scale__

    ---

    Determine optimal batch sizes by analyzing gradient noise. 
    Based on the theory that optimal batch size scales with gradient noise.

    [:octicons-arrow-right-24: Learn more](features/gradient-noise.md)

-   :material-microscope:{ .lg .middle } __Activation Probing__

    ---

    Discover where class information is encoded in your network by training 
    linear probes on intermediate activations.

    [:octicons-arrow-right-24: Learn more](features/activation-probing.md)

-   :material-shield-check:{ .lg .middle } __Pre-training Checks__

    ---

    Detect common issues before training: vanishing/exploding gradients, 
    poor initialization, dead neurons, and more.

    [:octicons-arrow-right-24: Learn more](features/precheck.md)

</div>

---

## 🚀 Quick Start

=== "Installation"

    ```bash
    git clone https://github.com/JonathanDray/tensight.git
    cd tensight
    pip install -r requirements.txt
    pip install -e .
    ```

=== "Basic Usage"

    ```python
    from tensight.analyzers import LossLandscapeAnalyzer
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Your model and data
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset, batch_size=64)

    # Analyze loss landscape
    analyzer = LossLandscapeAnalyzer(model, criterion, data_loader)
    results = analyzer.analyze(num_points=21, range_val=1.0)

    print(f"Sharpness: {results['sharpness']:.4f}")
    ```

=== "Example Output"

    ```
    🗺️ Analyzing Loss Landscape...
       Grid: 21x21
       Range: [-1.0, +1.0]
       Filter normalization: True
       ...
    
    Sharpness: 0.1350
    Interpretation: Flat (good generalization expected)
    ```

---

## 📚 Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    New to Tensight? Start here!

    - [Installation Guide](getting-started/installation.md)
    - [Quick Start Tutorial](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __Features__

    ---

    Explore all the powerful features

    - [Loss Landscape](features/loss-landscape.md)
    - [Gradient Noise](features/gradient-noise.md)
    - [Activation Probing](features/activation-probing.md)
    - [Pre-training Checks](features/precheck.md)

-   :material-code-tags:{ .lg .middle } __API Reference__

    ---

    Complete API documentation

    - [Analyzers](api/analyzers.md)
    - [Detectors](api/detectors.md)

-   :material-lightbulb-on:{ .lg .middle } __Examples__

    ---

    Real-world examples and tutorials

    - [MNIST Example](examples/mnist.md)
    - [CNN Example](examples/cnn.md)

</div>

---

## 🔬 Research Background

Tensight implements techniques from cutting-edge research:

- **Loss Landscape**: Based on ["Visualizing the Loss Landscape of Neural Nets"](https://arxiv.org/abs/1712.09913) (Li et al., NeurIPS 2018)
- **Gradient Noise**: Implements gradient noise scale theory for optimal batch sizing
- **Activation Probing**: Linear probing to understand representation learning

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](contributing.md) for details.

---

<div style="text-align: center; margin: 3rem 0; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
  <h3 style="color: white; margin-bottom: 1rem;">Ready to get started?</h3>
  <p style="color: rgba(255,255,255,0.9); margin-bottom: 1.5rem;">
    Install Tensight and start analyzing your models in minutes!
  </p>
  <a href="getting-started/installation.md" style="display: inline-block; padding: 0.75rem 2rem; background: white; color: #667eea; text-decoration: none; border-radius: 5px; font-weight: bold; transition: transform 0.2s;">
    Get Started →
  </a>
</div>

---

**Made with ❤️ for the deep learning community**
