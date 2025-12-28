# Analyzers

Advanced analysis tools for deep learning research.

## Available analyzers

### Loss Landscape
- `analyzers.loss_landscape.LossLandscapeAnalyzer`

### Gradient Noise
- `analyzers.gradient_noise.GradientNoiseAnalyzer`

### Hessian
- `analyzers.hessian.HessianAnalyzer`

### Fisher Information
- `analyzers.fisher.FisherAnalyzer`

### Neural Tangent Kernel
- `analyzers.ntk.NTKAnalyzer`

### Intrinsic Dimensionality
- `analyzers.intrinsic_dim.IntrinsicDimAnalyzer`

### Mode Connectivity
- `analyzers.mode_connectivity.ModeConnectivityAnalyzer`

### Lottery Ticket
- `analyzers.lottery_ticket.LotteryTicketAnalyzer`

### Influence Functions
- `analyzers.influence.InfluenceAnalyzer`

### Gradient Flow
- `analyzers.gradient_flow.GradientFlowAnalyzer`

## Usage

```python
from tensight.analyzers.loss_landscape import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(model, loss_fn, data_loader)
results = analyzer.analyze()
```

