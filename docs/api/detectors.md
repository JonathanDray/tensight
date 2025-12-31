# Detectors API Reference

## Gradient Detector

Detect vanishing or exploding gradients.

```python
from tensight.detectors import GradientDetector

detector = GradientDetector()
issues = detector.detect(model, data_loader)
```

## Learning Rate Detector

Detect learning rate issues.

```python
from tensight.detectors import LearningRateDetector

detector = LearningRateDetector()
issues = detector.detect(model, optimizer, data_loader)
```

## Dead Neurons Detector

Detect neurons that never activate.

```python
from tensight.detectors import DeadNeuronsDetector

detector = DeadNeuronsDetector()
issues = detector.detect(model, data_loader)
```

