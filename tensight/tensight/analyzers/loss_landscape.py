import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Callable

class LossLandscapeAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        data_loader: torch.utils.data.DataLoader
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.num_batches = 3
    
    def analyze(
        self,
        num_points: int = 21,
        range_val: float = 1.0,
        use_filter_norm: bool = True,
        num_batches: int = 3,
        seed: int = 42
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_batches = num_batches
        
        print("\n🗺️ Analyzing Loss Landscape...")
        print(f"   Filter normalization: {use_filter_norm}")
        print(f"   Grid: {num_points}x{num_points}")
        print(f"   Range: [-{range_val}, +{range_val}]")
        
        params_ref = self._get_params_vector()
        
        if use_filter_norm:
            dir1 = self._get_filter_normalized_direction()
            dir2 = self._get_filter_normalized_direction()
        else:
            dir1 = self._random_direction()
            dir2 = self._random_direction()
        
        dir2 = self._gram_schmidt(dir2, dir1)
        
        loss_center = self._compute_loss()
        print(f"   Center loss: {loss_center:.6f}")
        
        print(f"   Computing landscape...")
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
            "landscape_min": float(np.min(landscape_2d)),
            "landscape_max": float(np.max(landscape_2d)),
            "landscape_std": float(np.std(landscape_2d)),
            "range": range_val,
            "num_points": num_points,
            "filter_normalized": use_filter_norm,
        }
        
        self._print_results(results)
        return results
    
    def _get_params_vector(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1).clone() for p in self.model.parameters()])
    
    def _set_params_vector(self, params: torch.Tensor) -> None:
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(params[idx:idx + numel].view(p.shape))
            idx += numel
    
    def _random_direction(self) -> torch.Tensor:
        direction = torch.cat([torch.randn_like(p.data).view(-1) for p in self.model.parameters()])
        return direction / (direction.norm() + 1e-10)
    
    def _get_filter_normalized_direction(self) -> torch.Tensor:
        direction_parts = []
        
        for param in self.model.parameters():
            d = torch.randn_like(param.data)
            
            if param.dim() == 1:
                param_norm = param.data.norm().item()
                d_norm = d.norm().item()
                if param_norm > 1e-10 and d_norm > 1e-10:
                    d = d * (param_norm / d_norm)
            
            elif param.dim() == 2:
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm().item()
                    d_norm = d[i].norm().item()
                    if filter_norm > 1e-10 and d_norm > 1e-10:
                        d[i] = d[i] * (filter_norm / d_norm)
            
            elif param.dim() == 4:
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm().item()
                    d_norm = d[i].norm().item()
                    if filter_norm > 1e-10 and d_norm > 1e-10:
                        d[i] = d[i] * (filter_norm / d_norm)
            
            else:
                param_norm = param.data.norm().item()
                d_norm = d.norm().item()
                if param_norm > 1e-10 and d_norm > 1e-10:
                    d = d * (param_norm / d_norm)
            
            direction_parts.append(d.view(-1))
        
        return torch.cat(direction_parts)
    
    def _gram_schmidt(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        proj_coeff = torch.dot(v, u) / (torch.dot(u, u) + 1e-10)
        v_orth = v - proj_coeff * u
        v_norm = v_orth.norm()
        if v_norm > 1e-10:
            v_orth = v_orth / v_norm
        return v_orth
    
    def _compute_loss(self) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                if i >= self.num_batches:
                    break
                X = batch[0].to(self.device)
                y = batch[1].to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)
                total_loss += loss.item()
                count += 1
        
        return total_loss / max(count, 1)
    
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
        
        for i, a1 in enumerate(alphas):
            for j, a2 in enumerate(alphas):
                params_new = params_ref + a1 * dir1 + a2 * dir2
                self._set_params_vector(params_new)
                landscape[i, j] = self._compute_loss()
            
            progress = 100 * (i + 1) // num_points
            print(f"   Progress: {progress}%", end='\r')
        
        print()
        self._set_params_vector(params_ref)
        return landscape
    
    def _compute_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        return float(np.mean(landscape) - center_loss)
    
    def _compute_max_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        return float(np.max(landscape) - center_loss)
    
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
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("📊 Loss Landscape Results")
        print("=" * 50)
        print(f"   Center loss:       {results['center_loss']:.6f}")
        print(f"   Landscape min:     {results['landscape_min']:.6f}")
        print(f"   Landscape max:     {results['landscape_max']:.6f}")
        print(f"   Landscape std:     {results['landscape_std']:.6f}")
        print("-" * 50)
        print(f"   Avg sharpness:     {results['sharpness']:.6f}")
        print(f"   Max sharpness:     {results['sharpness_max']:.6f}")
        print(f"   Interpretation:    {results['sharpness_interpretation']}")
        print("-" * 50)
        print(f"   Filter normalized: {results['filter_normalized']}")
        print("=" * 50)
        
        if results['sharpness'] > 1.0:
            print("\n💡 Suggestions for flatter minimum:")
            print("   • Use SAM optimizer (Sharpness-Aware Minimization)")
            print("   • Increase batch size")
            print("   • Add regularization (dropout, weight decay)")
            print("   • Use learning rate warmup")
            print("   • Train longer with lower learning rate")