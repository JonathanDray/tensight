"""
Gradient Noise Scale Analyzer

Compute the gradient noise scale to find optimal batch size.

Based on: "An Empirical Model of Large-Batch Training"
          McCandlish et al., OpenAI, 2018
          https://arxiv.org/abs/1812.06162

Key insight: The gradient noise scale B_noise predicts the optimal batch size.
Training with batch_size ≈ B_noise is most compute-efficient.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Callable


class GradientNoiseAnalyzer:
    """
    Analyze gradient noise to determine optimal batch size.
    
    The gradient noise scale B_noise is defined as:
        B_noise = tr(Σ) / ||G||²
    
    Where:
        - Σ = covariance matrix of per-sample gradients
        - G = mean gradient
    
    If current batch_size < B_noise: increase batch size for efficiency
    If current batch_size > B_noise: decrease batch size (diminishing returns)
    
    Example:
        analyzer = GradientNoiseAnalyzer(model, loss_fn, train_loader)
        results = analyzer.analyze(num_batches=20)
        
        print(f"Current batch: {results['current_batch_size']}")
        print(f"Optimal batch: {results['optimal_batch_size']}")
        print(f"Efficiency: {results['efficiency']*100:.0f}%")
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
            model: PyTorch model
            loss_fn: Loss function
            data_loader: DataLoader for training data
        """
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = next(model.parameters()).device
    
    def analyze(self, num_batches: int = 20) -> Dict[str, Any]:
        """
        Compute gradient noise scale.
        
        Args:
            num_batches: Number of batches to sample gradients from
        
        Returns:
            Dictionary containing:
            - gradient_noise_scale: The B_noise value
            - current_batch_size: Batch size of data_loader
            - optimal_batch_size: Recommended batch size
            - efficiency: How efficient current batch size is (0-1)
            - mean_gradient_norm: ||G||
            - gradient_variance: tr(Σ)
        """
        print("\n📊 Analyzing Gradient Noise Scale...")
        
        # Collect gradients from multiple batches
        all_gradients = self._collect_gradients(num_batches)
        
        if len(all_gradients) < 2:
            print("   ⚠️ Not enough batches to compute noise scale")
            return {
                "gradient_noise_scale": float('inf'),
                "current_batch_size": 0,
                "optimal_batch_size": 32,
                "efficiency": 0.0,
            }
        
        # Compute noise scale
        noise_scale, mean_norm, variance = self._compute_noise_scale(all_gradients)
        
        # Get current batch size
        current_batch_size = next(iter(self.data_loader))[0].shape[0]
        
        # Estimate optimal batch size
        optimal_batch_size = self._estimate_optimal_batch_size(noise_scale)
        
        # Compute efficiency
        if noise_scale > 0 and noise_scale != float('inf'):
            efficiency = min(current_batch_size / noise_scale, 1.0)
        else:
            efficiency = 1.0
        
        results = {
            "gradient_noise_scale": noise_scale,
            "current_batch_size": current_batch_size,
            "optimal_batch_size": optimal_batch_size,
            "efficiency": efficiency,
            "mean_gradient_norm": mean_norm,
            "gradient_variance": variance,
        }
        
        self._print_results(results)
        
        return results
    
    def _collect_gradients(self, num_batches: int) -> List[torch.Tensor]:
        """Collect gradient vectors from multiple batches."""
        all_grads = []
        
        self.model.train()
        data_iter = iter(self.data_loader)
        
        for batch_idx in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator if we run out of batches
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
            
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            
            # Forward + backward
            self.model.zero_grad()
            output = self.model(X)
            loss = self.loss_fn(output, y)
            loss.backward()
            
            # Collect gradients as flat vector
            grad_vec = torch.cat([
                p.grad.view(-1).clone()
                for p in self.model.parameters()
                if p.grad is not None
            ])
            
            all_grads.append(grad_vec)
            
            print(f"   Collecting gradients: {batch_idx+1}/{num_batches}", end='\r')
        
        print()  # New line
        
        return all_grads
    
    def _compute_noise_scale(
        self,
        gradients: List[torch.Tensor]
    ) -> tuple:
        """
        Compute B_noise = tr(Σ) / ||G||²
        
        Args:
            gradients: List of gradient vectors from different batches
        
        Returns:
            Tuple of (noise_scale, mean_gradient_norm, variance)
        """
        # Stack gradients: [num_batches, num_params]
        grad_matrix = torch.stack(gradients)
        
        # Mean gradient G
        mean_grad = grad_matrix.mean(dim=0)
        mean_grad_norm_sq = (mean_grad ** 2).sum().item()
        mean_grad_norm = np.sqrt(mean_grad_norm_sq)
        
        if mean_grad_norm_sq < 1e-10:
            return float('inf'), mean_grad_norm, 0.0
        
        # Variance: tr(Σ) = E[||g - G||²]
        centered = grad_matrix - mean_grad
        variance = (centered ** 2).mean().item() * grad_matrix.shape[1]
        
        # Noise scale
        noise_scale = variance / mean_grad_norm_sq
        
        return noise_scale, mean_grad_norm, variance
    
    def _estimate_optimal_batch_size(self, noise_scale: float) -> int:
        """
        Estimate optimal batch size from noise scale.
        
        Optimal batch ≈ B_noise
        
        Args:
            noise_scale: Computed gradient noise scale
        
        Returns:
            Recommended batch size (power of 2)
        """
        if noise_scale == float('inf') or noise_scale <= 0:
            return 32  # Default
        
        # Optimal is approximately the noise scale
        optimal = int(noise_scale)
        
        # Clamp to reasonable range
        optimal = max(8, min(optimal, 4096))
        
        # Round to nearest power of 2
        log2_optimal = int(np.log2(max(optimal, 1)))
        optimal = 2 ** log2_optimal
        
        return optimal
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted results."""
        print("\n" + "=" * 50)
        print("📊 Gradient Noise Scale Results")
        print("=" * 50)
        print(f"   Gradient Noise Scale: {results['gradient_noise_scale']:.2f}")
        print(f"   Mean Gradient Norm:   {results['mean_gradient_norm']:.6f}")
        print(f"   Gradient Variance:    {results['gradient_variance']:.6f}")
        print("-" * 50)
        print(f"   Current batch size:   {results['current_batch_size']}")
        print(f"   Optimal batch size:   {results['optimal_batch_size']}")
        print(f"   Efficiency:           {results['efficiency']*100:.1f}%")
        print("=" * 50)
        
        # Suggestions
        efficiency = results['efficiency']
        current = results['current_batch_size']
        optimal = results['optimal_batch_size']
        
        if efficiency < 0.5:
            print(f"\n💡 Suggestion: Increase batch size to {optimal}")
            print(f"   You're wasting ~{(1-efficiency)*100:.0f}% of compute!")
        elif current > optimal * 2:
            print(f"\n💡 Suggestion: Decrease batch size to {optimal}")
            print(f"   Diminishing returns with current batch size.")
        else:
            print(f"\n✅ Batch size is reasonable!")
