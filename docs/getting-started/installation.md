# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy 1.21 or higher
- Matplotlib 3.5 or higher

## Install from Source

=== "Step 1: Clone Repository"

    ```bash
    git clone https://github.com/JonathanDray/tensight.git
    cd tensight
    ```

=== "Step 2: Create Virtual Environment"

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

=== "Step 3: Install Dependencies"

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Verify Installation

```python
import tensight
from tensight.analyzers import LossLandscapeAnalyzer
print("✅ Tensight installed successfully!")
```

## Dependencies

The main dependencies are:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0.0 | PyTorch deep learning framework |
| `torchvision` | >=0.15.0 | Vision datasets and transforms |
| `numpy` | >=1.21.0 | Numerical computations |
| `matplotlib` | >=3.5.0 | Visualizations |

## Development Installation

For development, you may also want to install:

```bash
pip install pytest  # For running tests
pip install mkdocs mkdocs-material  # For documentation
```

## Next Steps

Once installed, check out the [Quick Start Guide](quickstart.md) to run your first analysis!
