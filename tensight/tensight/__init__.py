"""
Tensight - Deep Learning Debugger
See through your models.

Usage:
    import tensight
    
    # Pre-check before training
    report = tensight.pre_check(model, (X, y), {'lr': 0.01})
    
    # Watch during training
    model = tensight.watch(model)
    model.record_loss(loss)
    model.diagnose()
"""

from .watcher import watch, WatchedModel
from .precheck import pre_check

__version__ = "0.1.0"
__author__ = "Tensight Team"

__all__ = [
    "watch",
    "WatchedModel",
    "pre_check",
]
