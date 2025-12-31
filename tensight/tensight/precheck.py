"""
Pre-Check module - Analyze model, data, and config before training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, Union
from torch.utils.data import DataLoader

from .report import Report, Problem


class PreCheck:
    """
    Pre-training analyzer.
    
    Checks for common issues before spending compute on training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pre-check analyzer.
        
        Args:
            model: PyTorch model to analyze
            data: Training data as (X, y) tuple or DataLoader
            config: Training config with 'lr', 'batch_size', 'epochs'
        """
        self.model = model
        self.data = data
        self.config = config or {}
        self.report = Report(model_name="pre-check")
        self.device = next(model.parameters()).device
    
    def run(self) -> Report:
        """
        Run all pre-checks.
        
        Returns:
            Report with findings
        """
        print("\n🔍 TENSIGHT PRE-CHECK")
        print("=" * 50)
        
        self._check_model()
        self._check_data()
        self._check_config()
        self._quick_test()
        self._estimate_cost()
        
        self.report.display()
        
        return self.report
    
    def _extract_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract X, y from various data formats."""
        
        if isinstance(self.data, tuple) and len(self.data) == 2:
            X, y = self.data
            if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
                return X, y
        
        elif isinstance(self.data, DataLoader):
            try:
                batch = next(iter(self.data))
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    return batch[0], batch[1]
            except StopIteration:
                pass
        
        elif isinstance(self.data, torch.Tensor):
            return self.data, None
        
        return None, None
    
    def _check_model(self) -> None:
        """Analyze model architecture."""
        
        print("\n📐 Analyzing model...")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.report.add_stat("total_params", total_params)
        self.report.add_stat("trainable_params", trainable_params)
        
        print(f"   Parameters: {total_params:,}")
        
        # Get data size for ratio
        X, _ = self._extract_data()
        if X is not None:
            data_size = X.shape[0]
            ratio = total_params / data_size
            
            if ratio > 100:
                self.report.add_problem(Problem(
                    name="Model Too Large",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"{total_params:,} params for {data_size:,} samples (ratio: {ratio:.0f}x)",
                    suggestion="High overfitting risk! Reduce model size, add more data, or use strong regularization",
                    details={"params": total_params, "samples": data_size, "ratio": ratio}
                ))
            elif ratio > 10:
                self.report.add_problem(Problem(
                    name="Model Possibly Too Large",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Params/samples ratio is {ratio:.0f}x",
                    suggestion="Consider dropout, weight decay, data augmentation, or early stopping",
                    details={"ratio": ratio}
                ))
        
        # Check weight initialization
        self._check_weights()
        
        # Count network depth
        depth = sum(1 for m in self.model.modules() 
                   if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)))
        self.report.add_stat("depth", depth)
        
        if depth > 10:
            # Check for skip connections
            has_skip = any('residual' in str(type(m)).lower() or 
                          'skip' in str(type(m)).lower() 
                          for m in self.model.modules())
            
            if not has_skip:
                self.report.add_problem(Problem(
                    name="Deep Network Without Skip Connections",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Network has {depth} layers without apparent skip connections",
                    suggestion="Consider adding residual/skip connections to help gradient flow",
                    details={"depth": depth}
                ))
    
    def _check_weights(self) -> None:
        """Check weight initialization."""
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name or param.dim() < 2:
                continue
            
            std = param.std().item()
            mean = param.mean().item()
            
            if std < 0.001:
                self.report.add_problem(Problem(
                    name="Weights Too Small",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Layer '{name}': std={std:.6f} (very small)",
                    suggestion="Use Xavier (for tanh/sigmoid) or Kaiming (for ReLU) initialization",
                    details={"layer": name, "std": std, "mean": mean},
                    paper_ref="He et al., 2015"
                ))
            
            elif std > 2.0:
                self.report.add_problem(Problem(
                    name="Weights Too Large",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Layer '{name}': std={std:.2f} (very large)",
                    suggestion="Gradients will explode! Reinitialize with proper scaling",
                    details={"layer": name, "std": std, "mean": mean}
                ))
    
    def _check_data(self) -> None:
        """Analyze training data."""
        
        print("\n📦 Analyzing data...")
        
        X, y = self._extract_data()
        
        if X is None:
            print("   ⚠️ Could not parse data format")
            return
        
        print(f"   X shape: {X.shape}")
        if y is not None:
            print(f"   y shape: {y.shape}")
        
        # Check for NaN
        nan_count = torch.isnan(X).sum().item()
        if nan_count > 0:
            self.report.add_problem(Problem(
                name="NaN in Data",
                severity=Problem.SEVERITY_ERROR,
                description=f"Found {nan_count:,} NaN values in input data",
                suggestion="Clean your data: X = X[~torch.isnan(X).any(dim=1)]",
                details={"nan_count": nan_count}
            ))
        
        # Check normalization
        X_float = X.float()
        mean = X_float.mean().item()
        std = X_float.std().item()
        
        if abs(mean) > 10 or std > 100:
            self.report.add_problem(Problem(
                name="Data Not Normalized",
                severity=Problem.SEVERITY_WARNING,
                description=f"Data stats: mean={mean:.2f}, std={std:.2f}",
                suggestion="Normalize data: X = (X - X.mean()) / X.std()",
                details={"mean": mean, "std": std}
            ))
        else:
            self.report.add_good(f"Data normalization OK (mean={mean:.2f}, std={std:.2f})")
        
        # Check class imbalance
        if y is not None:
            self._check_class_imbalance(y)
    
    def _check_class_imbalance(self, y: torch.Tensor) -> None:
        """Check for class imbalance."""
        
        # Handle one-hot encoded labels
        if y.dim() > 1:
            y = y.argmax(dim=1)
        
        unique, counts = torch.unique(y, return_counts=True)
        
        if len(unique) > 1:
            min_count = counts.min().item()
            max_count = counts.max().item()
            ratio = max_count / max(min_count, 1)
            
            if ratio > 10:
                self.report.add_problem(Problem(
                    name="Severe Class Imbalance",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Class ratio: {ratio:.0f}x (largest/smallest: {max_count}/{min_count})",
                    suggestion="Use class_weight, SMOTE oversampling, or focal loss",
                    details={"ratio": ratio, "min_count": min_count, "max_count": max_count}
                ))
            elif ratio > 3:
                self.report.add_problem(Problem(
                    name="Class Imbalance",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Class ratio: {ratio:.1f}x",
                    suggestion="Consider class_weight parameter or data augmentation",
                    details={"ratio": ratio}
                ))
    
    def _check_config(self) -> None:
        """Check training configuration."""
        
        print("\n⚙️ Analyzing config...")
        
        lr = self.config.get('lr') or self.config.get('learning_rate')
        batch_size = self.config.get('batch_size')
        epochs = self.config.get('epochs')
        
        if lr:
            print(f"   Learning rate: {lr}")
            
            if lr > 0.1:
                self.report.add_problem(Problem(
                    name="Learning Rate Very High",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"lr={lr} is dangerously high",
                    suggestion="Start with lr=0.001 for Adam or lr=0.01 for SGD",
                    details={"lr": lr}
                ))
            elif lr > 0.01:
                self.report.add_problem(Problem(
                    name="Learning Rate High",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"lr={lr} may cause instability",
                    suggestion="Try lr=0.001 if training is unstable",
                    details={"lr": lr}
                ))
        
        if batch_size:
            print(f"   Batch size: {batch_size}")
            
            if batch_size < 8:
                self.report.add_problem(Problem(
                    name="Batch Size Very Small",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"batch_size={batch_size} will have noisy gradients",
                    suggestion="Increase batch size to at least 16-32 for stable training",
                    details={"batch_size": batch_size}
                ))
        
        if epochs:
            print(f"   Epochs: {epochs}")
    
    def _quick_test(self) -> None:
        """Run quick forward/backward test."""
        
        print("\n🧪 Quick forward/backward test...")
        
        X, y = self._extract_data()
        if X is None:
            return
        
        try:
            # Use just 2 samples
            mini_X = X[:2].to(self.device)
            
            self.model.train()
            output = self.model(mini_X)
            
            print(f"   ✅ Forward OK - Output shape: {output.shape}")
            
            # Check for NaN in output
            if torch.isnan(output).any():
                self.report.add_problem(Problem(
                    name="NaN in Forward Pass",
                    severity=Problem.SEVERITY_ERROR,
                    description="Model produces NaN outputs",
                    suggestion="Check weight initialization and activation functions"
                ))
                return
            
            # Backward pass
            loss = output.sum()
            loss.backward()
            
            print(f"   ✅ Backward OK")
            
            # Check for NaN in gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    self.report.add_problem(Problem(
                        name="NaN in Backward Pass",
                        severity=Problem.SEVERITY_ERROR,
                        description=f"Layer '{name}' has NaN gradients",
                        suggestion="Check for numerical instability (log(0), division by zero)"
                    ))
                    break
            
            # Clear gradients
            self.model.zero_grad()
            
        except Exception as e:
            self.report.add_problem(Problem(
                name="Forward/Backward Test Failed",
                severity=Problem.SEVERITY_ERROR,
                description=str(e),
                suggestion="Fix this error before attempting to train"
            ))
    
    def _estimate_cost(self) -> None:
        """Estimate training time and cost."""
        
        print("\n💰 Estimating cost...")
        
        X, _ = self._extract_data()
        data_size = X.shape[0] if X is not None else 10000
        
        params = sum(p.numel() for p in self.model.parameters())
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        # Rough FLOPS estimation
        # Forward: ~2 * params per sample
        # Backward: ~4 * params per sample
        # Total: ~6 * params per sample
        flops_per_sample = 6 * params
        total_flops = flops_per_sample * data_size * epochs
        
        # Assume ~10 TFLOPS GPU (like RTX 3080)
        gpu_tflops = 10e12
        estimated_seconds = total_flops / gpu_tflops
        estimated_hours = estimated_seconds / 3600
        
        # Cloud GPU cost: ~$1.50/hour
        estimated_cost = estimated_hours * 1.50
        
        self.report.add_stat("estimated_hours", round(estimated_hours, 2))
        self.report.add_stat("estimated_cost_usd", round(estimated_cost, 2))
        
        print(f"   ⏱️ Estimated time: {estimated_hours:.1f}h")
        print(f"   💵 Estimated cost: ${estimated_cost:.2f}")
        
        if estimated_cost > 10:
            self.report.add_problem(Problem(
                name="Expensive Training",
                severity=Problem.SEVERITY_INFO,
                description=f"Training will cost ~${estimated_cost:.2f}",
                suggestion="Make sure your config is correct before starting!"
            ))


def pre_check(
    model: nn.Module,
    data: Union[Tuple[torch.Tensor, torch.Tensor], DataLoader],
    config: Optional[Dict[str, Any]] = None
) -> Report:
    """
    Run pre-training checks on model, data, and config.
    
    Args:
        model: PyTorch model to analyze
        data: Training data as (X, y) tuple or DataLoader
        config: Training config with 'lr', 'batch_size', 'epochs'
    
    Returns:
        Report with findings
    
    Example:
        report = tensight.pre_check(
            model,
            (X_train, y_train),
            {'lr': 0.001, 'batch_size': 32, 'epochs': 100}
        )
        
        if not report.can_train:
            print("Fix issues before training!")
    """
    checker = PreCheck(model, data, config)
    return checker.run()
