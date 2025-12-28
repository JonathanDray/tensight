# tensight

Analysis and diagnostics tool for PyTorch models.

## Project Structure

```
tensight/
│
├── tensight/
│   ├── __init__.py                    # Entry point
│   ├── watcher.py                     # Wrapper that observes the model
│   ├── precheck.py                    # Pre-training checks
│   ├── autofix.py                     # Automatic corrections
│   │
│   ├── report/
│   │   ├── __init__.py
│   │   ├── builder.py                 # Report generation
│   │   ├── visualizer.py              # Graphs and plots
│   │   └── export.py                  # Export to PDF/LaTeX/HTML
│   │
│   ├── detectors/                     # Basic detectors
│   │   ├── __init__.py
│   │   ├── gradient.py                # Vanishing/Exploding gradients
│   │   ├── learning_rate.py           # Learning rate issues
│   │   ├── dead_neurons.py            # Dead neurons
│   │   ├── activation.py              # Activation issues
│   │   ├── initialization.py          # Poor weight initialization
│   │   └── numerical.py               # NaN, Inf, underflow/overflow
│   │
│   ├── analyzers/                     # Advanced analyzers (research)
│   │   ├── __init__.py
│   │   ├── loss_landscape.py          # Loss Landscape Analysis
│   │   ├── gradient_noise.py          # Gradient Noise Scale
│   │   ├── hessian.py                 # Hessian Eigenspectrum
│   │   ├── fisher.py                  # Fisher Information Matrix
│   │   ├── lottery_ticket.py          # Lottery Ticket / Pruning
│   │   ├── intrinsic_dim.py           # Intrinsic Dimensionality
│   │   ├── mode_connectivity.py       # Mode Connectivity
│   │   ├── ntk.py                     # Neural Tangent Kernel
│   │   ├── influence.py               # Influence Functions
│   │   └── gradient_flow.py           # Advanced Gradient Flow
│   │
│   ├── data/                          # Data analysis
│   │   ├── __init__.py
│   │   ├── quality.py                 # NaN, outliers, duplicates
│   │   ├── distribution.py            # Class imbalance, stats
│   │   ├── labeling.py                # Incorrect label detection
│   │   └── augmentation.py            # Augmentation suggestions
│   │
│   ├── architecture/                  # Architecture analysis
│   │   ├── __init__.py
│   │   ├── bottleneck.py              # Bottleneck detection
│   │   ├── redundancy.py              # Redundant layers/neurons
│   │   └── suggestions.py             # Improvement suggestions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── hooks.py                   # PyTorch hooks utilities
│       ├── math.py                    # Mathematical computations
│       └── hardware.py                # GPU/VRAM detection
│
├── tests/
│   ├── test_watcher.py
│   ├── test_precheck.py
│   ├── test_detectors.py
│   └── test_analyzers.py
│
├── examples/
│   ├── basic_usage.py
│   ├── precheck_demo.py
│   ├── loss_landscape_demo.py
│   └── full_analysis.py
│
├── setup.py                           # pip installation
├── pyproject.toml                     # Modern config
├── README.md
├── LICENSE
└── requirements.txt
```

## Installation

```bash
pip install -e .
```

## Usage

```python
import tensight

# Wrap your model
model = tensight.watch(model)
```
