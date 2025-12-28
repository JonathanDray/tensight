import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np


class GradientNoiseAnalyzer:
    
    
    def __init__(self, model: nn.Module, loss_fn, data_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
    
    def analyze(self, num_batches: int = 10) -> Dict[str, Any]:
        
        
        print("\n📊 Analyzing Gradient Noise Scale...")
        
        
        all_gradients = self._collect_gradients(num_batches)
        
        
        gradient_noise_scale = self._compute_noise_scale(all_gradients)
        
        
        current_batch_size = next(iter(self.data_loader))[0].shape[0]
        
        
        optimal_batch_size = self._estimate_optimal_batch_size(
            gradient_noise_scale, 
            current_batch_size
        )
        
        results = {
            "gradient_noise_scale": gradient_noise_scale,
            "current_batch_size": current_batch_size,
            "optimal_batch_size": optimal_batch_size,
            "efficiency": min(current_batch_size / optimal_batch_size, 1.0),
        }
        
        self._print_results(results)
        
        return results
    
    def _collect_gradients(self, num_batches: int) -> List[torch.Tensor]:
        
        
        all_grads = []
        
        self.model.train()
        data_iter = iter(self.data_loader)
        
        for _ in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
            
            X, y = batch[0], batch[1]
            
            
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss_fn(output, y)
            loss.backward()
            
            
            grad_vec = torch.cat([
                p.grad.view(-1) for p in self.model.parameters()
                if p.grad is not None
            ])
            all_grads.append(grad_vec.clone())
        
        return all_grads
    
    def _compute_noise_scale(self, gradients: List[torch.Tensor]) -> float:
        
        
        
        grad_matrix = torch.stack(gradients)  
        
        
        mean_grad = grad_matrix.mean(dim=0)
        mean_grad_norm_sq = (mean_grad ** 2).sum().item()
        
        if mean_grad_norm_sq < 1e-10:
            return float('inf')
        
        
        centered = grad_matrix - mean_grad
        variance = (centered ** 2).mean(dim=0).sum().item()
        
        
        noise_scale = variance / mean_grad_norm_sq
        
        return noise_scale
    
    def _estimate_optimal_batch_size(
        self, 
        noise_scale: float,
        current_batch_size: int
    ) -> int:
        
        
        if noise_scale == float('inf'):
            return current_batch_size
        
        
        optimal = int(noise_scale)
        
        
        optimal = max(8, min(optimal, 4096))
        
        
        optimal = 2 ** int(np.log2(optimal))
        
        return optimal
    
    def _print_results(self, results: Dict[str, Any]):
        
        
        print("\n📊 Gradient Noise Scale Results:")
        print("-" * 40)
        print(f"   Noise Scale: {results['gradient_noise_scale']:.2f}")
        print(f"   Current batch size: {results['current_batch_size']}")
        print(f"   Optimal batch size: {results['optimal_batch_size']}")
        print(f"   Efficiency: {results['efficiency']*100:.1f}%")
        
        if results['efficiency'] < 0.5:
            print("\n   💡 Suggestions:")
            print(f"      • Increase batch size to {results['optimal_batch_size']}")
            print(f"      • You're losing ~{(1-results['efficiency'])*100:.0f}% compute efficiency")