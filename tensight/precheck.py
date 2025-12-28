import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .report import Report, Problem


class PreCheck:
    
    
    def __init__(self, model: nn.Module, data, config: Dict[str, Any] = None):
        self.model = model
        self.data = data
        self.config = config or {}
        self.report = Report()
    
    def run(self) -> Report:
        
        
        print("\n🔍 TENSIGHT PRE-CHECK")
        print("=" * 50)
        
        self._check_model()
        self._check_data()
        self._check_config()
        self._quick_test()
        self._estimate_cost()
        
        self.report.display()
        return self.report
    
    def _check_model(self):
        
        
        print("\n📐 Analyzing model...")
        
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.report.add_stat("total_params", total_params)
        self.report.add_stat("trainable_params", trainable)
        
        print(f"   Parameters: {total_params:,}")
        
        
        X, _ = self._extract_data()
        if X is not None:
            data_size = len(X)
            ratio = total_params / data_size
            
            if ratio > 100:
                self.report.add_problem(Problem(
                    name="Model Too Large",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"{total_params:,} params for {data_size:,} samples (ratio: {ratio:.0f}x)",
                    suggestion="High overfitting risk. Reduce model size or get more data."
                ))
            elif ratio > 10:
                self.report.add_problem(Problem(
                    name="Model Possibly Too Large",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"High params/samples ratio ({ratio:.0f}x)",
                    suggestion="Use dropout, data augmentation, or early stopping"
                ))
        
        
        self._check_weights()
    
    def _check_weights(self):
        
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name or param.dim() < 2:
                continue
            
            std = param.std().item()
            
            if std < 0.001:
                self.report.add_problem(Problem(
                    name="Weights Too Small",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Layer '{name}': std={std:.6f}",
                    suggestion="Use Xavier or Kaiming initialization"
                ))
            elif std > 2.0:
                self.report.add_problem(Problem(
                    name="Weights Too Large",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Layer '{name}': std={std:.2f}",
                    suggestion="Gradients will explode. Reinitialize weights."
                ))
    
    def _check_data(self):
        
        
        print("\n📦 Analyzing data...")
        
        X, y = self._extract_data()
        
        if X is None:
            print("   ⚠️ Could not parse data format")
            return
        
        print(f"   X shape: {X.shape}")
        if y is not None:
            print(f"   y shape: {y.shape}")
        
        
        if torch.isnan(X).any():
            nan_count = torch.isnan(X).sum().item()
            self.report.add_problem(Problem(
                name="NaN in Data",
                severity=Problem.SEVERITY_ERROR,
                description=f"{nan_count} NaN values found in X",
                suggestion="Clean your data before training"
            ))
        
        
        mean = X.float().mean().item()
        std = X.float().std().item()
        
        if abs(mean) > 10 or std > 100:
            self.report.add_problem(Problem(
                name="Data Not Normalized",
                severity=Problem.SEVERITY_WARNING,
                description=f"mean={mean:.2f}, std={std:.2f}",
                suggestion="Normalize data (mean=0, std=1) for stable training"
            ))
        
        
        if y is not None:
            self._check_class_imbalance(y)
    
    def _check_class_imbalance(self, y: torch.Tensor):
        
        
        if y.dim() > 1:
            y = y.argmax(dim=1)
        
        unique, counts = torch.unique(y, return_counts=True)
        
        if len(unique) > 1:
            ratios = counts.float() / counts.sum()
            max_ratio = ratios.max().item()
            min_ratio = ratios.min().item()
            
            if max_ratio / min_ratio > 10:
                self.report.add_problem(Problem(
                    name="Severe Class Imbalance",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"Max/min ratio: {max_ratio/min_ratio:.0f}x",
                    suggestion="Use class_weight, oversampling (SMOTE), or focal loss"
                ))
            elif max_ratio / min_ratio > 3:
                self.report.add_problem(Problem(
                    name="Class Imbalance",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"Max/min ratio: {max_ratio/min_ratio:.1f}x",
                    suggestion="Consider class_weight or data augmentation"
                ))
    
    def _check_config(self):
        
        
        print("\n⚙️ Analyzing config...")
        
        lr = self.config.get('lr', self.config.get('learning_rate'))
        batch_size = self.config.get('batch_size')
        
        if lr:
            print(f"   Learning rate: {lr}")
            if lr > 0.1:
                self.report.add_problem(Problem(
                    name="Learning Rate Very High",
                    severity=Problem.SEVERITY_ERROR,
                    description=f"lr={lr} is dangerous",
                    suggestion="Start with lr=0.001 or lower"
                ))
            elif lr > 0.01:
                self.report.add_problem(Problem(
                    name="Learning Rate High",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"lr={lr} may cause instability",
                    suggestion="Try lr=0.001 if training is unstable"
                ))
        
        if batch_size:
            print(f"   Batch size: {batch_size}")
            if batch_size < 8:
                self.report.add_problem(Problem(
                    name="Batch Size Very Small",
                    severity=Problem.SEVERITY_WARNING,
                    description=f"batch_size={batch_size}",
                    suggestion="Gradients will be noisy. Increase if possible."
                ))
    
    def _quick_test(self):
        
        
        print("\n🧪 Quick forward/backward test...")
        
        X, _ = self._extract_data()
        if X is None:
            return
        
        try:
            mini_X = X[:2] if len(X) > 2 else X
            self.model.train()
            output = self.model(mini_X)
            
            print(f"   ✅ Forward OK - Output: {output.shape}")
            
            if torch.isnan(output).any():
                self.report.add_problem(Problem(
                    name="NaN in Output",
                    severity=Problem.SEVERITY_ERROR,
                    description="Forward pass produces NaN",
                    suggestion="Check activations and weight initialization"
                ))
            
            loss = output.sum()
            loss.backward()
            print(f"   ✅ Backward OK")
            
            self.model.zero_grad()
            
        except Exception as e:
            self.report.add_problem(Problem(
                name="Test Failed",
                severity=Problem.SEVERITY_ERROR,
                description=str(e),
                suggestion="Fix this error before training"
            ))
    
    def _estimate_cost(self):
        
        
        print("\n💰 Estimating time/cost...")
        
        params = sum(p.numel() for p in self.model.parameters())
        epochs = self.config.get('epochs', 100)
        
        X, _ = self._extract_data()
        data_size = len(X) if X is not None else 10000
        
        
        flops_per_sample = 2 * params * 3
        total_flops = flops_per_sample * data_size * epochs
        
        gpu_tflops = 10e12
        estimated_hours = (total_flops / gpu_tflops) / 3600
        estimated_cost = estimated_hours * 1.5
        
        self.report.add_stat("estimated_hours", round(estimated_hours, 2))
        self.report.add_stat("estimated_cost_usd", round(estimated_cost, 2))
        
        print(f"   ⏱️ Estimated time: {estimated_hours:.1f}h")
        print(f"   💵 Estimated cost: ${estimated_cost:.2f}")
    
    def _extract_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        if isinstance(self.data, tuple) and len(self.data) == 2:
            return self.data[0], self.data[1]
        elif isinstance(self.data, torch.utils.data.DataLoader):
            batch = next(iter(self.data))
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                return batch[0], batch[1]
        elif isinstance(self.data, torch.Tensor):
            return self.data, None
        
        return None, None


def pre_check(model: nn.Module, data, config: Dict[str, Any] = None) -> Report:
    
    checker = PreCheck(model, data, config)
    return checker.run()