import torch
from typing import List, Dict
from ..report import Problem


class GradientDetector:
    
    
    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 1000
    
    def detect(
        self,
        gradients: Dict[str, torch.Tensor],
        activations: Dict[str, torch.Tensor] = None,
        loss_history: List[float] = None,
        model: torch.nn.Module = None
    ) -> List[Problem]:
        
        problems = []
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            
            
            if torch.isnan(grad).any():
                problems.append(Problem(
                    name="NaN Gradient",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Layer '{name}' has NaN gradients",
                    suggestion="Reduce learning rate or use gradient clipping"
                ))
            
            
            elif grad_mean < self.VANISHING_THRESHOLD:
                problems.append(Problem(
                    name="Vanishing Gradient",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Layer '{name}': very small gradients (mean={grad_mean:.2e})",
                    suggestion="Use ReLU/GELU instead of Sigmoid/Tanh, or add BatchNorm",
                    paper_ref="Glorot & Bengio, 2010"
                ))
            
            
            elif grad_max > self.EXPLODING_THRESHOLD:
                problems.append(Problem(
                    name="Exploding Gradient",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Layer '{name}': huge gradients (max={grad_max:.2e})",
                    suggestion="Use gradient clipping (max_norm=1.0) or reduce LR",
                    paper_ref="Pascanu et al., 2013"
                ))
        
        return problems