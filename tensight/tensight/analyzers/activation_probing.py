"""
Activation Probing Analyzer

Discover what information is encoded at each layer of your model.

Based on: "Understanding intermediate layers using linear classifier probes"
          Alain & Bengio, 2016
          
          "What do Neural Machine Translation Models Learn about Morphology?"
          Belinkov et al., 2017

Key insight: Train simple linear classifiers on intermediate activations.
If the probe achieves high accuracy, that information is encoded at that layer.

Works with: CNNs, MLPs, Transformers, any PyTorch model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import OrderedDict


class LinearProbe(nn.Module):
    """Simple linear classifier for probing."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        # Flatten if needed (for CNN feature maps)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


class ActivationProber:
    """
    Probe what information is encoded at each layer.
    
    Usage:
        prober = ActivationProber(model)
        
        # Probe specific layers
        results = prober.probe(
            train_loader=train_loader,
            test_loader=test_loader,
            target_fn=lambda batch: batch[1],  # labels
            layer_names=['layer1', 'layer2', 'fc']
        )
        
        # Results show accuracy per layer
        # High accuracy = information is encoded there
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize prober.
        
        Args:
            model: Any PyTorch model
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.activations = {}
        self.hooks = []
    
    def get_layer_names(self) -> List[str]:
        """Get all named modules in the model."""
        return [name for name, _ in self.model.named_modules() if name]
    
    def _register_hook(self, layer_name: str):
        """Register forward hook to capture activations."""
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_name] = output.detach()
        
        # Find the layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                return True
        return False
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _collect_activations(
        self,
        data_loader: torch.utils.data.DataLoader,
        layer_names: List[str],
        target_fn: callable,
        max_samples: int = 2000
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collect activations and targets from data.
        
        Returns:
            Dict with 'activations' and 'targets' per layer
        """
        # Register hooks
        for name in layer_names:
            if not self._register_hook(name):
                print(f"   ⚠️ Layer '{name}' not found")
        
        collected = {name: {'activations': [], 'targets': []} for name in layer_names}
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if total_samples >= max_samples:
                    break
                
                # Get input and target
                X = batch[0].to(self.device)
                targets = target_fn(batch).to(self.device)
                
                # Forward pass (triggers hooks)
                self.activations = {}
                _ = self.model(X)
                
                # Collect activations
                for name in layer_names:
                    if name in self.activations:
                        act = self.activations[name]
                        # Flatten spatial dimensions for CNNs
                        if act.dim() > 2:
                            act = act.view(act.size(0), -1)
                        collected[name]['activations'].append(act.cpu())
                        collected[name]['targets'].append(targets.cpu())
                
                total_samples += X.size(0)
        
        self._remove_hooks()
        
        # Concatenate
        for name in layer_names:
            if collected[name]['activations']:
                collected[name]['activations'] = torch.cat(collected[name]['activations'])
                collected[name]['targets'] = torch.cat(collected[name]['targets'])
        
        return collected
    
    def _train_probe(
        self,
        train_acts: torch.Tensor,
        train_targets: torch.Tensor,
        test_acts: torch.Tensor,
        test_targets: torch.Tensor,
        num_classes: int,
        epochs: int = 20,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """Train a linear probe and return metrics."""
        
        input_dim = train_acts.shape[1]
        probe = LinearProbe(input_dim, num_classes).to(self.device)
        
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        train_acts = train_acts.to(self.device)
        train_targets = train_targets.to(self.device)
        
        probe.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = probe(train_acts)
            loss = criterion(output, train_targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            # Train accuracy
            train_preds = probe(train_acts).argmax(dim=1)
            train_acc = (train_preds == train_targets).float().mean().item()
            
            # Test accuracy
            test_acts = test_acts.to(self.device)
            test_targets = test_targets.to(self.device)
            test_preds = probe(test_acts).argmax(dim=1)
            test_acc = (test_preds == test_targets).float().mean().item()
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'input_dim': input_dim
        }
    
    def probe(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        target_fn: callable = lambda batch: batch[1],
        layer_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None,
        max_samples: int = 2000,
        probe_epochs: int = 20
    ) -> Dict[str, Any]:
        """
        Probe layers to discover what information is encoded.
        
        Args:
            train_loader: DataLoader for training probes
            test_loader: DataLoader for evaluating probes
            target_fn: Function to extract targets from batch (default: batch[1])
            layer_names: List of layer names to probe (default: auto-detect)
            num_classes: Number of classes (default: auto-detect)
            max_samples: Max samples to use for probing
            probe_epochs: Epochs to train each probe
        
        Returns:
            Dictionary with probe results per layer
        """
        print("\n🔬 Activation Probing Analysis")
        print("=" * 50)
        
        # Auto-detect layers if not specified
        if layer_names is None:
            all_layers = self.get_layer_names()
            # Filter to interesting layers (skip containers)
            layer_names = [
                name for name in all_layers
                if not any(skip in name.lower() for skip in ['sequential', 'modulelist'])
            ]
            print(f"   Auto-detected {len(layer_names)} layers")
        
        print(f"   Probing {len(layer_names)} layers...")
        print(f"   Max samples: {max_samples}")
        
        # Collect activations
        print("\n📦 Collecting activations (train)...")
        train_data = self._collect_activations(
            train_loader, layer_names, target_fn, max_samples
        )
        
        print("📦 Collecting activations (test)...")
        test_data = self._collect_activations(
            test_loader, layer_names, target_fn, max_samples // 2
        )
        
        # Auto-detect num_classes
        if num_classes is None:
            for name in layer_names:
                if train_data[name]['targets'].numel() > 0:
                    num_classes = int(train_data[name]['targets'].max().item()) + 1
                    break
        print(f"   Classes: {num_classes}")
        
        # Train probes
        print("\n🎯 Training probes...")
        results = {}
        
        for name in layer_names:
            if train_data[name]['activations'].numel() == 0:
                continue
            
            probe_results = self._train_probe(
                train_data[name]['activations'],
                train_data[name]['targets'],
                test_data[name]['activations'],
                test_data[name]['targets'],
                num_classes=num_classes,
                epochs=probe_epochs
            )
            
            results[name] = probe_results
            acc = probe_results['test_accuracy']
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"   {name:<20} {bar} {acc*100:.1f}%")
        
        # Summary
        self._print_summary(results)
        
        return {
            'layer_results': results,
            'num_classes': num_classes,
            'num_layers_probed': len(results)
        }
    
    def _print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print summary of probing results."""
        if not results:
            print("\n⚠️ No results to summarize")
            return
        
        # Find best and worst layers
        sorted_layers = sorted(
            results.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        best_layer, best_results = sorted_layers[0]
        worst_layer, worst_results = sorted_layers[-1]
        
        print("\n" + "=" * 50)
        print("📊 Probing Summary")
        print("=" * 50)
        print(f"   Best layer:  {best_layer} ({best_results['test_accuracy']*100:.1f}%)")
        print(f"   Worst layer: {worst_layer} ({worst_results['test_accuracy']*100:.1f}%)")
        print("-" * 50)
        print("   💡 Interpretation:")
        print(f"      • High accuracy = information IS encoded")
        print(f"      • Low accuracy = information NOT encoded yet")
        print(f"      • Rising accuracy = information being FORMED")
        print("=" * 50)


def probe_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to probe a model.
    
    Usage:
        results = probe_model(model, train_loader, test_loader)
    """
    prober = ActivationProber(model)
    return prober.probe(train_loader, test_loader, layer_names=layer_names)
