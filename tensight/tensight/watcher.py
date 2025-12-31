"""
Watcher module - Monitor models during training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .detectors import run_all_detectors
from .report import Report


class WatchedModel(nn.Module):
    """
    Wrapper that monitors a model during training.
    
    Captures gradients, activations, and loss history for analysis.
    
    Usage:
        model = WatchedModel(model)
        # or
        model = tensight.watch(model)
        
        # Train normally
        for epoch in range(epochs):
            loss = train_step(model, data)
            model.record_loss(loss)
        
        # Get diagnosis
        report = model.diagnose()
    """
    
    def __init__(self, model: nn.Module, name: str = "model"):
        """
        Wrap a model for monitoring.
        
        Args:
            model: PyTorch model to wrap
            name: Name for the model (used in reports)
        """
        super().__init__()
        
        self.model = model
        self.name = name
        
        # Storage for analysis
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.loss_history: List[float] = []
        
        # Hook handles for cleanup
        self._forward_hooks = []
        self._backward_hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on all layers."""
        
        for name, module in self.model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
            
            # Forward hook for activations
            handle = module.register_forward_hook(self._make_forward_hook(name))
            self._forward_hooks.append(handle)
            
            # Backward hook for gradients (on parameters)
            if hasattr(module, 'weight') and module.weight is not None:
                handle = module.weight.register_hook(self._make_backward_hook(name))
                self._backward_hooks.append(handle)
    
    def _make_forward_hook(self, name: str):
        """Create forward hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Store detached copy to avoid memory issues
                self.activations[name] = output.detach().clone()
        return hook
    
    def _make_backward_hook(self, name: str):
        """Create backward hook to capture gradients."""
        def hook(grad):
            if grad is not None:
                self.gradients[name] = grad.detach().clone()
        return hook
    
    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)
    
    def record_loss(self, loss: Union[torch.Tensor, float]) -> None:
        """
        Record a loss value for analysis.
        
        Args:
            loss: Loss value (tensor or float)
        """
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = float(loss)
        
        self.loss_history.append(loss_value)
    
    def diagnose(self, verbose: bool = True) -> Report:
        """
        Run all detectors and generate diagnostic report.
        
        Args:
            verbose: Whether to print the report
        
        Returns:
            Report object with all findings
        """
        report = Report(model_name=self.name)
        
        # Add basic stats
        total_params = sum(p.numel() for p in self.model.parameters())
        report.add_stat("total_parameters", total_params)
        report.add_stat("loss_recordings", len(self.loss_history))
        report.add_stat("layers_monitored", len(self.activations))
        
        if self.loss_history:
            report.add_stat("initial_loss", self.loss_history[0])
            report.add_stat("final_loss", self.loss_history[-1])
        
        # Run all detectors
        problems = run_all_detectors(
            gradients=self.gradients,
            activations=self.activations,
            loss_history=self.loss_history,
            model=self.model
        )
        
        report.add_problems(problems)
        
        # Add good things if no problems
        if not problems:
            report.add_good("Gradient flow is healthy")
            report.add_good("No dead neurons detected")
            report.add_good("Loss is decreasing normally")
        
        if verbose:
            report.display()
        
        return report
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.gradients.clear()
        self.activations.clear()
        self.loss_history.clear()
    
    def remove_hooks(self) -> None:
        """Remove all hooks (call when done monitoring)."""
        for handle in self._forward_hooks:
            handle.remove()
        for handle in self._backward_hooks:
            handle.remove()
        
        self._forward_hooks.clear()
        self._backward_hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def watch(model: nn.Module, name: str = "model") -> WatchedModel:
    """
    Wrap a model for monitoring during training.
    
    Args:
        model: PyTorch model to wrap
        name: Name for the model (used in reports)
    
    Returns:
        WatchedModel wrapper
    
    Example:
        model = tensight.watch(model)
        
        for epoch in range(epochs):
            loss = train_step(model, data)
            model.record_loss(loss)
        
        model.diagnose()
    """
    return WatchedModel(model, name=name)
