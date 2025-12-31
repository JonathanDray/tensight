"""
Dead Neurons Detector - Detects neurons that always output zero.
"""

import torch
from typing import List, Dict, Optional
import torch.nn as nn

from ..report import Problem


class DeadNeuronsDetector:
    """
    Detects dead neurons (always zero output).
    
    Common with ReLU activation when neurons get stuck in negative region.
    """
    
    # Threshold: more than 50% zeros = dead
    DEAD_THRESHOLD = 0.5
    
    def detect(
        self,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        loss_history: Optional[List[float]] = None,
        model: Optional[nn.Module] = None
    ) -> List[Problem]:
        """
        Analyze activations for dead neurons.
        
        Args:
            gradients: Not used by this detector
            activations: Dictionary of layer_name -> activation tensor
            loss_history: Not used by this detector
            model: Not used by this detector
        
        Returns:
            List of detected problems
        """
        problems = []
        
        if not activations:
            return problems
        
        for name, activation in activations.items():
            if activation is None:
                continue
            
            # Calculate ratio of zeros
            total_elements = activation.numel()
            zero_elements = (activation == 0).sum().item()
            zero_ratio = zero_elements / total_elements
            
            if zero_ratio > self.DEAD_THRESHOLD:
                problems.append(Problem(
                    name="Dead Neurons",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Layer '{name}': {zero_ratio*100:.1f}% of neurons are dead (always zero)",
                    suggestion="Use Leaky ReLU (nn.LeakyReLU(0.01)) instead of ReLU, or reduce learning rate",
                    details={
                        "layer": name,
                        "zero_ratio": zero_ratio,
                        "zero_count": zero_elements,
                        "total_count": total_elements
                    },
                    paper_ref="Maas et al., 2013 (Leaky ReLU)"
                ))
        
        return problems
