import torch
import torch.nn as nn
from .detectors import run_all_detectors
from .report import Report


class WatchedModel(nn.Module):
    
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.report = Report()
        
        
        self.gradients = {}
        self.activations = {}
        self.loss_history = []
        
        
        self._register_hooks()
    
    def _register_hooks(self):
        
        
        for name, layer in self.model.named_modules():
            
            layer.register_forward_hook(self._save_activation(name))
            
            
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.register_hook(self._save_gradient(name))
    
    def _save_activation(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
        return hook
    
    def _save_gradient(self, name: str):
        def hook(grad):
            self.gradients[name] = grad.detach()
        return hook
    
    def forward(self, x):
        return self.model(x)
    
    def record_loss(self, loss):
        
        if isinstance(loss, torch.Tensor):
            self.loss_history.append(loss.item())
        else:
            self.loss_history.append(float(loss))
    
    def diagnose(self) -> Report:
        
        
        problems = run_all_detectors(
            gradients=self.gradients,
            activations=self.activations,
            loss_history=self.loss_history,
            model=self.model
        )
        
        self.report.add_problems(problems)
        self.report.display()
        
        return self.report


def watch(model: nn.Module) -> WatchedModel:
    
    return WatchedModel(model)