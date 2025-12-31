"""
Test Gradient Noise Scale - MNIST
Trouve le batch size optimal pour ton modèle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from tensight.analyzers.gradient_noise import GradientNoiseAnalyzer


# ============ MODEL ============

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ============ UTILS ============

def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)


def train_epoch(model, loader, optimizer):
    """Train pour une epoch, retourne loss moyenne"""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    
    for X, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader):
    """Retourne accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += len(y)
    return 100 * correct / total


def test_batch_sizes(batch_sizes, epochs=3):
    """Compare différents batch sizes"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=512)
    
    results = {}
    
    for bs in batch_sizes:
        print(f"\n📦 Testing batch_size={bs}")
        
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        losses = []
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer)
            losses.append(loss)
            print(f"   Epoch {epoch+1}: Loss={loss:.4f}")
        
        acc = evaluate(model, test_loader)
        steps_per_epoch = len(train_loader)
        
        results[bs] = {
            'final_loss': losses[-1],
            'test_acc': acc,
            'steps_per_epoch': steps_per_epoch,
            'total_steps': steps_per_epoch * epochs,
            'losses': losses
        }
        print(f"   Test Acc: {acc:.1f}% | Steps/epoch: {steps_per_epoch}")
    
    return results


def plot_results(noise_results, batch_results, filename):
    """Visualisation des résultats"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    batch_sizes = list(batch_results.keys())
    
    # 1. Test Accuracy vs Batch Size
    ax = axes[0, 0]
    accs = [batch_results[bs]['test_acc'] for bs in batch_sizes]
    ax.plot(batch_sizes, accs, 'b-o', lw=2, markersize=8)
    ax.axvline(noise_results['optimal_batch_size'], color='g', ls='--', lw=2, label=f"Optimal: {noise_results['optimal_batch_size']}")
    ax.axvline(noise_results['current_batch_size'], color='r', ls=':', lw=2, label=f"Current: {noise_results['current_batch_size']}")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs Batch Size")
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Steps to Train vs Batch Size
    ax = axes[0, 1]
    steps = [batch_results[bs]['total_steps'] for bs in batch_sizes]
    ax.plot(batch_sizes, steps, 'r-s', lw=2, markersize=8)
    ax.axvline(noise_results['optimal_batch_size'], color='g', ls='--', lw=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Total Steps (3 epochs)")
    ax.set_title("Training Steps vs Batch Size")
    ax.set_xscale('log', base=2)
    ax.grid(alpha=0.3)
    
    # 3. Efficiency curve
    ax = axes[1, 0]
    noise_scale = noise_results['gradient_noise_scale']
    bs_range = np.logspace(np.log2(8), np.log2(512), 50, base=2)
    efficiency = np.minimum(bs_range / noise_scale, 1.0)
    ax.plot(bs_range, efficiency * 100, 'g-', lw=2)
    ax.axvline(noise_results['optimal_batch_size'], color='g', ls='--', lw=2, label=f"Optimal: {noise_results['optimal_batch_size']}")
    ax.axhline(100, color='gray', ls=':', alpha=0.5)
    ax.fill_between(bs_range, 0, efficiency * 100, alpha=0.2, color='green')
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title(f"Compute Efficiency (B_noise = {noise_scale:.0f})")
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
╔═══════════════════════════════════════════════════╗
║         GRADIENT NOISE SCALE ANALYSIS             ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  Gradient Noise Scale (B_noise): {noise_results['gradient_noise_scale']:<15.1f} ║
║  Mean Gradient Norm:             {noise_results['mean_gradient_norm']:<15.6f} ║
║                                                   ║
╠═══════════════════════════════════════════════════╣
║  Current Batch Size:   {noise_results['current_batch_size']:<10}               ║
║  Optimal Batch Size:   {noise_results['optimal_batch_size']:<10}               ║
║  Current Efficiency:   {noise_results['efficiency']*100:<10.1f}%              ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  BATCH SIZE COMPARISON (3 epochs)                 ║
║  ─────────────────────────────────                ║"""
    
    for bs in batch_sizes:
        r = batch_results[bs]
        summary += f"""
║  BS={bs:<4} → Acc={r['test_acc']:.1f}%  Steps={r['total_steps']:<5}        ║"""
    
    summary += f"""
║                                                   ║
╠═══════════════════════════════════════════════════╣
║  💡 RECOMMENDATION                                ║
║                                                   ║"""
    
    if noise_results['efficiency'] < 0.5:
        summary += f"""
║  Increase batch size to {noise_results['optimal_batch_size']} for             ║
║  {(1/noise_results['efficiency']):.1f}x faster training!                      ║"""
    else:
        summary += f"""
║  Current batch size is efficient!                 ║"""
    
    summary += """
║                                                   ║
╚═══════════════════════════════════════════════════╝
    """
    
    ax.text(0.0, 0.95, summary, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved: {filename}")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("📊 Gradient Noise Scale Analysis - MNIST")
    print("=" * 50)
    
    # Analyze with current batch size
    current_batch = 32
    print(f"\n📦 Loading MNIST (batch_size={current_batch})...")
    train_loader = load_mnist(batch_size=current_batch)
    
    # Create model and do a few training steps first
    print("\n🏋️ Quick pre-training (1 epoch)...")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss = train_epoch(model, train_loader, optimizer)
    print(f"   Loss after 1 epoch: {loss:.4f}")
    
    # Analyze gradient noise
    print("\n" + "-" * 50)
    loss_fn = nn.CrossEntropyLoss()
    analyzer = GradientNoiseAnalyzer(model, loss_fn, train_loader)
    noise_results = analyzer.analyze(num_batches=30)
    
    # Test different batch sizes
    print("\n" + "-" * 50)
    print("🧪 Comparing batch sizes...")
    batch_sizes = [16, 32, 64, 128, 256]
    batch_results = test_batch_sizes(batch_sizes, epochs=3)
    
    # Plot
    print("\n" + "-" * 50)
    print("📈 Generating visualization...")
    plot_results(noise_results, batch_results, 'gradient_noise_report.png')
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"""
    Gradient Noise Scale: {noise_results['gradient_noise_scale']:.1f}
    
    Current batch:  {current_batch}
    Optimal batch:  {noise_results['optimal_batch_size']}
    Efficiency:     {noise_results['efficiency']*100:.0f}%
    
    Batch Size | Test Acc | Steps
    -----------|----------|-------""")
    
    for bs in batch_sizes:
        r = batch_results[bs]
        marker = " ← optimal" if bs == noise_results['optimal_batch_size'] else ""
        print(f"    {bs:<6}   | {r['test_acc']:.1f}%   | {r['total_steps']:<5}{marker}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()