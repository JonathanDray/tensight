"""
Learning Rate Detector - Detects LR problems from loss history.
"""

import torch
from typing import List, Dict, Optional
import torch.nn as nn

from ..report import Problem


class LearningRateDetector:
    """
    Detects learning rate problems by analyzing loss history.
    
    Problems detected:
    - Loss explosion (LR too high)
    - Loss oscillation (LR too high)
    - Loss stagnation (LR too low or stuck)
    """
    
    # Minimum history length to analyze
    MIN_HISTORY_LENGTH = 5
    
    def detect(
        self,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        loss_history: Optional[List[float]] = None,
        model: Optional[nn.Module] = None
    ) -> List[Problem]:
        """
        Analyze loss history for LR problems.
        
        Args:
            gradients: Not used by this detector
            activations: Not used by this detector
            loss_history: List of loss values over training
            model: Not used by this detector
        
        Returns:
            List of detected problems
        """
        problems = []
        
        if loss_history is None or len(loss_history) < self.MIN_HISTORY_LENGTH:
            return problems
        
        # Filter out NaN/Inf values
        clean_history = [l for l in loss_history if not (l != l or abs(l) == float('inf'))]
        if len(clean_history) < self.MIN_HISTORY_LENGTH:
            return problems
        
        # Check for explosion
        problems.extend(self._check_explosion(clean_history))
        
        # Check for oscillation
        problems.extend(self._check_oscillation(clean_history))
        
        # Check for stagnation
        problems.extend(self._check_stagnation(clean_history))
        
        return problems
    
    def _check_explosion(self, loss_history: List[float]) -> List[Problem]:
        """Check if loss is exploding."""
        problems = []
        
        initial_loss = loss_history[0]
        current_loss = loss_history[-1]
        
        # Loss increased by 10x or more
        if initial_loss > 0 and current_loss > initial_loss * 10:
            problems.append(Problem(
                name="Loss Explosion",
                severity=Problem.SEVERITY_ERROR,
                description=f"Loss exploded from {initial_loss:.4f} to {current_loss:.4f} ({current_loss/initial_loss:.1f}x increase)",
                suggestion="Reduce learning rate by 10x (e.g., lr /= 10)",
                details={
                    "initial_loss": initial_loss,
                    "current_loss": current_loss,
                    "increase_factor": current_loss / initial_loss
                }
            ))
        
        return problems
    
    def _check_oscillation(self, loss_history: List[float]) -> List[Problem]:
        """Check if loss is oscillating too much."""
        problems = []
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(loss_history)):
            prev_direction = loss_history[i-1] - loss_history[i-2]
            curr_direction = loss_history[i] - loss_history[i-1]
            
            # Direction changed
            if prev_direction * curr_direction < 0:
                direction_changes += 1
        
        # More than 60% direction changes = too much oscillation
        oscillation_ratio = direction_changes / (len(loss_history) - 2)
        
        if oscillation_ratio > 0.6:
            problems.append(Problem(
                name="Unstable Loss (Oscillation)",
                severity=Problem.SEVERITY_WARNING,
                description=f"Loss is oscillating too much ({direction_changes} direction changes, {oscillation_ratio*100:.0f}%)",
                suggestion="Reduce learning rate or increase batch size for smoother gradients",
                details={
                    "direction_changes": direction_changes,
                    "oscillation_ratio": oscillation_ratio
                }
            ))
        
        return problems
    
    def _check_stagnation(self, loss_history: List[float]) -> List[Problem]:
        """Check if loss has stagnated."""
        problems = []
        
        if len(loss_history) < 10:
            return problems
        
        # Compare recent loss to loss 10 steps ago
        old_loss = loss_history[-10]
        current_loss = loss_history[-1]
        
        if old_loss > 0:
            relative_change = abs(current_loss - old_loss) / old_loss
            
            # Less than 1% change and loss is still significant
            if relative_change < 0.01 and current_loss > 0.01:
                problems.append(Problem(
                    name="Stagnant Loss",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Loss barely changed in last 10 steps ({relative_change*100:.2f}% change)",
                    suggestion="Try increasing learning rate, using a different optimizer (Adam), or adding LR scheduler",
                    details={
                        "old_loss": old_loss,
                        "current_loss": current_loss,
                        "relative_change": relative_change
                    }
                ))
        
        return problems
