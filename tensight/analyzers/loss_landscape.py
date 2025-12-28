import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Callable
import copy


class LossLandscapeAnalyzer:
    
    
    def __init__(self, model: nn.Module, loss_fn: Callable, data_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
    
    def analyze(
        self, 
        num_points: int = 21, 
        range_val: float = 1.0,
        use_filter_norm: bool = True,
        num_batches: int = 3
    ) -> Dict[str, Any]:
        
        
        print("\n🗺️ Analyzing Loss Landscape...")
        if use_filter_norm:
            print("   Using filter-wise normalization (Li et al., 2018)")
        
        self.num_batches = num_batches
        
        
        params_ref = self._get_params_vector()
        
        
        if use_filter_norm:
            dir1 = self._get_filter_normalized_direction()
            dir2 = self._get_filter_normalized_direction()
        else:
            dir1 = self._random_direction()
            dir2 = self._random_direction()
        
        
        dir2 = self._gram_schmidt(dir2, dir1)
        
        
        loss_center = self._compute_loss()
        print(f"   Center loss: {loss_center:.4f}")
        
        
        print(f"   Computing {num_points}x{num_points} landscape...")
        landscape_2d = self._compute_2d_landscape(
            params_ref, dir1, dir2, num_points, range_val
        )
        
        
        sharpness = self._compute_sharpness(landscape_2d, loss_center)
        sharpness_max = self._compute_max_sharpness(landscape_2d, loss_center)
        
        results = {
            "center_loss": loss_center,
            "sharpness": sharpness,
            "sharpness_max": sharpness_max,
            "sharpness_interpretation": self._interpret_sharpness(sharpness),
            "landscape_2d": landscape_2d,
            "range": range_val,
            "num_points": num_points,
            "filter_normalized": use_filter_norm,
        }
        
        self._print_results(results)
        
        return results
    
    def _get_params_vector(self) -> torch.Tensor:
        
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _set_params_vector(self, params: torch.Tensor):
        
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = params[idx:idx + numel].view(p.shape).to(self.device)
            idx += numel
    
    def _random_direction(self) -> torch.Tensor:
        
        direction = torch.cat([
            torch.randn_like(p.data).view(-1)
            for p in self.model.parameters()
        ])
        direction = direction / direction.norm()
        return direction
    
    def _get_filter_normalized_direction(self) -> torch.Tensor:
        
        
        direction_parts = []
        
        for param in self.model.parameters():
            
            d = torch.randn_like(param.data)
            
            if param.dim() == 1:
                
                d = d * (param.data.norm() / (d.norm() + 1e-10))
            
            elif param.dim() == 2:
                
                
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm()
                    d_norm = d[i].norm() + 1e-10
                    d[i] = d[i] * (filter_norm / d_norm)
            
            elif param.dim() == 4:
                
                
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm()
                    d_norm = d[i].norm() + 1e-10
                    d[i] = d[i] * (filter_norm / d_norm)
            
            else:
                
                d = d * (param.data.norm() / (d.norm() + 1e-10))
            
            direction_parts.append(d.view(-1))
        
        direction = torch.cat(direction_parts)
        direction = direction / direction.norm()
        
        return direction
    
    def _gram_schmidt(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        
        v = v - (torch.dot(v, u) / torch.dot(u, u)) * u
        return v / v.norm()
    
    def _compute_loss(self) -> float:
        
        self.model.eval()
        
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if i >= self.num_batches:
                    break
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _compute_2d_landscape(
        self,
        params_ref: torch.Tensor,
        dir1: torch.Tensor,
        dir2: torch.Tensor,
        num_points: int,
        range_val: float
    ) -> np.ndarray:
        
        
        alphas = np.linspace(-range_val, range_val, num_points)
        landscape = np.zeros((num_points, num_points))
        
        total = num_points * num_points
        computed = 0
        
        for i, a1 in enumerate(alphas):
            for j, a2 in enumerate(alphas):
                params_new = params_ref + a1 * dir1 + a2 * dir2
                self._set_params_vector(params_new)
                landscape[i, j] = self._compute_loss()
                
                computed += 1
                if computed % (total // 5) == 0:
                    print(f"   Progress: {100*computed//total}%")
        
        
        self._set_params_vector(params_ref)
        
        return landscape
    
    def _compute_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        
        return np.mean(landscape) - center_loss
    
    def _compute_max_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        
        return np.max(landscape) - center_loss
    
    def _interpret_sharpness(self, sharpness: float) -> str:
        if sharpness < 0.01:
            return "Extremely flat (excellent generalization)"
        elif sharpness < 0.1:
            return "Very flat (good generalization)"
        elif sharpness < 0.5:
            return "Flat (good generalization expected)"
        elif sharpness < 1.0:
            return "Moderate (okay generalization)"
        elif sharpness < 5.0:
            return "Sharp (poor generalization likely)"
        else:
            return "Very sharp (overfitting likely)"
    
    def _print_results(self, results: Dict[str, Any]):
        print("\n📊 Loss Landscape Results:")
        print("-" * 45)
        print(f"   Center loss:       {results['center_loss']:.4f}")
        print(f"   Avg sharpness:     {results['sharpness']:.4f}")
        print(f"   Max sharpness:     {results['sharpness_max']:.4f}")
        print(f"   Interpretation:    {results['sharpness_interpretation']}")
        print(f"   Filter normalized: {results['filter_normalized']}")