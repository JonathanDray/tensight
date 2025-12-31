"""
Loss Landscape Analyzer

Visualize and analyze the geometry of the loss function around trained parameters.

Based on: "Visualizing the Loss Landscape of Neural Nets" 
          Li et al., NeurIPS 2018
          https://arxiv.org/abs/1712.09913

Key insight: Sharp minima tend to generalize poorly, flat minima generalize well.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Callable, Optional


class LossLandscapeAnalyzer:
    """
    Analyze the loss landscape around current model parameters.
    
    Features:
    - Filter-wise normalization (from original paper)
    - 2D landscape visualization
    - Sharpness metrics
    - Reproducible results
    
    Example:
        analyzer = LossLandscapeAnalyzer(model, loss_fn, train_loader)
        results = analyzer.analyze(num_points=21, range_val=1.0)
        
        print(f"Sharpness: {results['sharpness']}")
        # Plot results['landscape_2d'] with matplotlib
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        data_loader: torch.utils.data.DataLoader
    ):
        """
        Initialize analyzer.
        
        Args:
            model: Trained PyTorch model
            loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
            data_loader: DataLoader for computing loss
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
        self.num_batches = 3  # Default batches for loss computation
    
    def analyze(
        self,
        num_points: int = 21,
        range_val: float = 1.0,
        use_filter_norm: bool = True,
        num_batches: int = 3,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Compute loss landscape and sharpness metrics.
        
        Args:
            num_points: Grid resolution (num_points x num_points)
            range_val: Perturbation range [-range_val, +range_val]
            use_filter_norm: Use filter-wise normalization (recommended)
            num_batches: Number of batches to average loss over
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing:
            - center_loss: Loss at current parameters
            - sharpness: Average sharpness metric
            - sharpness_max: Maximum sharpness
            - sharpness_interpretation: Human-readable interpretation
            - landscape_2d: 2D numpy array of loss values
            - landscape_min/max/std: Statistics
            - filter_normalized: Whether filter norm was used
        """
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.num_batches = num_batches
        
        print("\n🗺️ Analyzing Loss Landscape...")
        print(f"   Filter normalization: {use_filter_norm}")
        print(f"   Grid: {num_points}x{num_points}")
        print(f"   Range: [-{range_val}, +{range_val}]")
        
        # Store reference parameters
        params_ref = self._get_params_vector()
        
        # Generate random directions
        if use_filter_norm:
            dir1 = self._get_filter_normalized_direction()
            dir2 = self._get_filter_normalized_direction()
        else:
            dir1 = self._random_direction()
            dir2 = self._random_direction()
        
        # Make directions orthogonal using Gram-Schmidt
        dir2 = self._gram_schmidt(dir2, dir1)
        
        # Compute center loss
        loss_center = self._compute_loss()
        print(f"   Center loss: {loss_center:.6f}")
        
        # Compute 2D landscape
        print(f"   Computing landscape...")
        landscape_2d = self._compute_2d_landscape(
            params_ref, dir1, dir2, num_points, range_val
        )
        
        # Compute metrics
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
        """Flatten all model parameters into a single vector."""
        return torch.cat([
            p.data.view(-1).clone() 
            for p in self.model.parameters()
        ])
    
    def _set_params_vector(self, params: torch.Tensor) -> None:
        """Set model parameters from a flat vector."""
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(params[idx:idx + numel].view(p.shape))
            idx += numel
    
    def _random_direction(self) -> torch.Tensor:
        """Generate a random unit direction (no filter normalization)."""
        direction = torch.cat([
            torch.randn_like(p.data).view(-1)
            for p in self.model.parameters()
        ])
        return direction / (direction.norm() + 1e-10)
    
    def _get_filter_normalized_direction(self) -> torch.Tensor:
        """
        Generate filter-wise normalized random direction.
        
        From Li et al., 2018:
        "We normalize each filter in d to have the same norm as the 
        corresponding filter in θ. This makes the direction meaningful
        and allows fair comparison across different models."
        
        For each parameter tensor:
        - 1D (bias): Scale whole vector
        - 2D (linear): Normalize each output neuron (row)
        - 4D (conv): Normalize each output channel
        """
        direction_parts = []
        
        for param in self.model.parameters():
            d = torch.randn_like(param.data)
            
            if param.dim() == 1:
                # Bias vector: scale to match parameter norm
                param_norm = param.data.norm().item()
                d_norm = d.norm().item()
                if param_norm > 1e-10 and d_norm > 1e-10:
                    d = d * (param_norm / d_norm)
            
            elif param.dim() == 2:
                # Linear layer: [out_features, in_features]
                # Normalize each output neuron (row) independently
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm().item()
                    d_norm = d[i].norm().item()
                    if filter_norm > 1e-10 and d_norm > 1e-10:
                        d[i] = d[i] * (filter_norm / d_norm)
            
            elif param.dim() == 4:
                # Conv layer: [out_channels, in_channels, H, W]
                # Normalize each output filter independently
                for i in range(param.shape[0]):
                    filter_norm = param.data[i].norm().item()
                    d_norm = d[i].norm().item()
                    if filter_norm > 1e-10 and d_norm > 1e-10:
                        d[i] = d[i] * (filter_norm / d_norm)
            
            else:
                # Other tensor shapes: scale whole tensor
                param_norm = param.data.norm().item()
                d_norm = d.norm().item()
                if param_norm > 1e-10 and d_norm > 1e-10:
                    d = d * (param_norm / d_norm)
            
            direction_parts.append(d.view(-1))
        
        # Concatenate WITHOUT normalizing (preserve filter-wise scaling)
        return torch.cat(direction_parts)
    
    def _gram_schmidt(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Make v orthogonal to u using Gram-Schmidt process."""
        # Project v onto u
        proj_coeff = torch.dot(v, u) / (torch.dot(u, u) + 1e-10)
        # Subtract projection
        v_orth = v - proj_coeff * u
        # Normalize
        v_norm = v_orth.norm()
        if v_norm > 1e-10:
            v_orth = v_orth / v_norm
        return v_orth
    
    def _compute_loss(self) -> float:
        """Compute loss averaged over multiple batches."""
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
        """
        Compute 2D loss landscape on a grid.
        
        Args:
            params_ref: Reference parameters (current trained model)
            dir1: First random direction
            dir2: Second random direction (orthogonal to dir1)
            num_points: Number of points per dimension
            range_val: Range to explore [-range_val, +range_val]
        
        Returns:
            2D numpy array of loss values
        """
        alphas = np.linspace(-range_val, range_val, num_points)
        landscape = np.zeros((num_points, num_points))
        
        for i, a1 in enumerate(alphas):
            for j, a2 in enumerate(alphas):
                # Perturb parameters: θ + α₁d₁ + α₂d₂
                params_new = params_ref + a1 * dir1 + a2 * dir2
                self._set_params_vector(params_new)
                
                # Compute loss
                landscape[i, j] = self._compute_loss()
            
            # Progress update
            progress = 100 * (i + 1) // num_points
            print(f"   Progress: {progress}%", end='\r')
        
        print()  # New line after progress
        
        # Restore original parameters
        self._set_params_vector(params_ref)
        
        return landscape
    
    def _compute_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        """
        Compute average sharpness.
        
        Sharpness = E[L(θ + d)] - L(θ)
        
        Higher sharpness indicates a sharper minimum.
        """
        return float(np.mean(landscape) - center_loss)
    
    def _compute_max_sharpness(self, landscape: np.ndarray, center_loss: float) -> float:
        """
        Compute maximum sharpness.
        
        Max Sharpness = max[L(θ + d)] - L(θ)
        """
        return float(np.max(landscape) - center_loss)
    
    def _interpret_sharpness(self, sharpness: float) -> str:
        """Convert sharpness value to human-readable interpretation."""
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
        """Print formatted results to console."""
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
        
        # Print suggestions for sharp minima
        if results['sharpness'] > 1.0:
            print("\n💡 Suggestions for flatter minimum:")
            print("   • Use SAM optimizer (Sharpness-Aware Minimization)")
            print("   • Increase batch size")
            print("   • Add regularization (dropout, weight decay)")
            print("   • Use learning rate warmup")
            print("   • Train longer with lower learning rate")
