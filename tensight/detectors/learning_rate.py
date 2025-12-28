from typing import List, Dict
import torch
from ..report import Problem


class LearningRateDetector:
    
    
    def detect(
        self,
        gradients: Dict[str, torch.Tensor] = None,
        activations: Dict[str, torch.Tensor] = None,
        loss_history: List[float] = None,
        model: torch.nn.Module = None
    ) -> List[Problem]:
        
        problems = []
        
        if loss_history is None or len(loss_history) < 5:
            return problems
        
        
        if loss_history[-1] > loss_history[0] * 10:
            problems.append(Problem(
                name="Learning Rate Too High",
                severity=Problem.SEVERITY_ERROR,
                description=f"Loss exploding: {loss_history[0]:.4f} → {loss_history[-1]:.4f}",
                suggestion="Divide learning rate by 10"
            ))
        
        
        oscillations = self._count_oscillations(loss_history)
        if oscillations > len(loss_history) * 0.6:
            problems.append(Problem(
                name="Unstable Loss",
                severity=Problem.SEVERITY_WARNING,
                description=f"Loss oscillating too much ({oscillations} direction changes)",
                suggestion="Reduce LR or increase batch size"
            ))
        
        
        if len(loss_history) >= 10:
            recent_change = abs(loss_history[-1] - loss_history[-10]) / (abs(loss_history[-10]) + 1e-8)
            if recent_change < 0.01 and loss_history[-1] > 0.01:
                problems.append(Problem(
                    name="Stagnant Loss",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Loss not decreasing (change: {recent_change*100:.2f}%)",
                    suggestion="Increase LR, try Adam optimizer, or add LR scheduler"
                ))
        
        return problems
    
    def _count_oscillations(self, loss_history: List[float]) -> int:
        oscillations = 0
        for i in range(2, len(loss_history)):
            prev_dir = loss_history[i-1] - loss_history[i-2]
            curr_dir = loss_history[i] - loss_history[i-1]
            if prev_dir * curr_dir < 0:
                oscillations += 1
        return oscillations