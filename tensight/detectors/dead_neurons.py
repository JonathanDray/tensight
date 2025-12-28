import torch
from typing import List, Dict
from ..report import Problem


class DeadNeuronsDetector:
    
    
    DEAD_THRESHOLD = 0.5
    
    def detect(
        self,
        gradients: Dict[str, torch.Tensor] = None,
        activations: Dict[str, torch.Tensor] = None,
        loss_history: List[float] = None,
        model: torch.nn.Module = None
    ) -> List[Problem]:
        
        problems = []
        
        if activations is None:
            return problems
        
        for name, activation in activations.items():
            if activation is None:
                continue
            
            zero_ratio = (activation == 0).float().mean().item()
            
            if zero_ratio > self.DEAD_THRESHOLD:
                problems.append(Problem(
                    name="Dead Neurons",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Layer '{name}': {zero_ratio*100:.1f}% neurons are dead",
                    suggestion="Use Leaky ReLU (negative_slope=0.01) or reduce LR",
                    paper_ref="Maas et al., 2013"
                ))
        
        return problems