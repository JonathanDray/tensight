"""
Detectors module - Fast checks for common training problems.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .gradient import GradientDetector
from .learning_rate import LearningRateDetector
from .dead_neurons import DeadNeuronsDetector
from ..report import Problem


# All available detectors
ALL_DETECTORS = [
    GradientDetector,
    LearningRateDetector,
    DeadNeuronsDetector,
]


def run_all_detectors(
    gradients: Dict[str, torch.Tensor],
    activations: Dict[str, torch.Tensor],
    loss_history: List[float],
    model: nn.Module
) -> List[Problem]:
    """
    Run all detectors and collect problems.
    
    Args:
        gradients: Dictionary of layer_name -> gradient tensor
        activations: Dictionary of layer_name -> activation tensor
        loss_history: List of loss values over training
        model: The PyTorch model
    
    Returns:
        List of detected problems
    """
    problems = []
    
    for DetectorClass in ALL_DETECTORS:
        detector = DetectorClass()
        detected = detector.detect(
            gradients=gradients,
            activations=activations,
            loss_history=loss_history,
            model=model
        )
        problems.extend(detected)
    
    return problems


__all__ = [
    "run_all_detectors",
    "GradientDetector",
    "LearningRateDetector",
    "DeadNeuronsDetector",
]
