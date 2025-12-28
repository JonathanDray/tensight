from .gradient import GradientDetector
from .learning_rate import LearningRateDetector
from .dead_neurons import DeadNeuronsDetector

ALL_DETECTORS = [
    GradientDetector,
    LearningRateDetector,
    DeadNeuronsDetector,
]


def run_all_detectors(gradients, activations, loss_history, model):
    
    problems = []
    
    for DetectorClass in ALL_DETECTORS:
        detector = DetectorClass()
        problems.extend(
            detector.detect(
                gradients=gradients,
                activations=activations,
                loss_history=loss_history,
                model=model
            )
        )
    
    return problems


__all__ = ["run_all_detectors"]