"""
Gradient Detector - Detects vanishing, exploding, and NaN gradients.
"""

import torch
from typing import List, Dict, Optional
import torch.nn as nn

from ..report import Problem


class GradientDetector:
    """
    Detects gradient-related problems.
    
    Problems detected:
    - NaN gradients
    - Infinite gradients
    - Vanishing gradients (mean < 1e-7)
    - Exploding gradients (max > 1000)
    """
    
    # Thresholds
    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 1000.0
    
    def detect(
        self,
        gradients: Dict[str, torch.Tensor],
        activations: Optional[Dict[str, torch.Tensor]] = None,
        loss_history: Optional[List[float]] = None,
        model: Optional[nn.Module] = None
    ) -> List[Problem]:
        """
        Analyze gradients for problems.
        
        Args:
            gradients: Dictionary of layer_name -> gradient tensor
            activations: Not used by this detector
            loss_history: Not used by this detector
            model: Not used by this detector
        
        Returns:
            List of detected problems
        """
        problems = []
        
        if not gradients:
            return problems
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            layer_problems = self._check_gradient(name, grad)
            problems.extend(layer_problems)
        
        return problems
    
    def _check_gradient(self, name: str, grad: torch.Tensor) -> List[Problem]:
        """Check a single gradient tensor."""
        problems = []
        
        # Compute statistics
        grad_abs = grad.abs()
        grad_mean = grad_abs.mean().item()
        grad_max = grad_abs.max().item()
        grad_std = grad.std().item()
        
        # Check for NaN
        nan_count = torch.isnan(grad).sum().item()
        if nan_count > 0:
            problems.append(Problem(
                name="NaN Gradient",
                severity=Problem.SEVERITY_ERROR,
                description=f"Layer '{name}' has {nan_count} NaN gradient values",
                suggestion="Reduce learning rate, use gradient clipping, or check for log(0)/division by zero",
                details={"layer": name, "nan_count": nan_count}
            ))
            return problems  # No point checking further if NaN
        
        # Check for Inf
        inf_count = torch.isinf(grad).sum().item()
        if inf_count > 0:
            problems.append(Problem(
                name="Infinite Gradient",
                severity=Problem.SEVERITY_ERROR,
                description=f"Layer '{name}' has {inf_count} infinite gradient values",
                suggestion="Reduce learning rate or use gradient clipping (max_norm=1.0)",
                details={"layer": name, "inf_count": inf_count}
            ))
            return problems
        
        # Check for vanishing gradients
        if grad_mean < self.VANISHING_THRESHOLD:
            problems.append(Problem(
                name="Vanishing Gradient",
                severity=Problem.SEVERITY_WARNING,
                description=f"Layer '{name}': gradients are very small (mean={grad_mean:.2e})",
                suggestion="Use ReLU/GELU instead of Sigmoid/Tanh, add BatchNorm, or use residual connections",
                details={"layer": name, "mean": grad_mean, "std": grad_std},
                paper_ref="Glorot & Bengio, 2010"
            ))
        
        # Check for exploding gradients
        elif grad_max > self.EXPLODING_THRESHOLD:
            problems.append(Problem(
                name="Exploding Gradient",
                severity=Problem.SEVERITY_ERROR,
                description=f"Layer '{name}': gradients are very large (max={grad_max:.2e})",
                suggestion="Use gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                details={"layer": name, "max": grad_max},
                paper_ref="Pascanu et al., 2013"
            ))
        
        return problems
