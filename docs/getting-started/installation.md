# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy 1.21 or higher
- Matplotlib 3.5 or higher

## Install from Source

```bash
# Clone the repository
git clone https://github.com/JonathanDray/tensight.git
cd tensight

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Verify Installation

```python
import tensight
from tensight.analyzers import LossLandscapeAnalyzer
print("Tensight installed successfully!")
```

## Dependencies

The main dependencies are:

- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Vision datasets and transforms
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Visualizations

## Development Installation

For development, you may also want to install:

```bash
pip install pytest  # For running tests
pip install mkdocs mkdocs-material  # For documentation
```

