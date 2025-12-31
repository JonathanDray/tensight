"""
Test Loss Landscape - MNIST
Compare Sharp vs Flat model landscapes with train/test accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from tensight.analyzers.loss_landscape import LossLandscapeAnalyzer


# ============ MODELS ============

class SharpModel(nn.Module):
    """Sans régularisation → minimum sharp"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class FlatModel(nn.Module):
    """Avec BatchNorm + Dropout → minimum flat"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ============ UTILS ============

def load_mnist(batch_size=64):
    """Charger MNIST train et test"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, test_loader


def evaluate(model, loader):
    """Calculer loss et accuracy"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            total_loss += loss_fn(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    
    return total_loss / len(loader), 100 * correct / total


def train(model, train_loader, test_loader, lr, epochs=5):
    """Entraînement avec suivi train/test"""
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        for X, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        
        # Evaluate
        train_loss, train_acc = evaluate(model, train_loader)
        test_loss, test_acc = evaluate(model, test_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"  Epoch {epoch+1}: Train={train_acc:.1f}% Test={test_acc:.1f}% (gap={train_acc-test_acc:.1f}%)")
    
    return history


def plot_results(sharp_res, flat_res, sharp_hist, flat_hist):
    """Visualisation comparative avec 3D et courbes train/test"""
    fig = plt.figure(figsize=(16, 14))
    
    # Row 1: Training curves
    ax1 = fig.add_subplot(3, 3, 1)
    epochs = range(1, len(sharp_hist['train_acc']) + 1)
    ax1.plot(epochs, sharp_hist['train_acc'], 'r-', lw=2, label='Sharp Train')
    ax1.plot(epochs, sharp_hist['test_acc'], 'r--', lw=2, label='Sharp Test')
    ax1.plot(epochs, flat_hist['train_acc'], 'b-', lw=2, label='Flat Train')
    ax1.plot(epochs, flat_hist['test_acc'], 'b--', lw=2, label='Flat Test')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Train vs Test Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(epochs, sharp_hist['train_loss'], 'r-', lw=2, label='Sharp Train')
    ax2.plot(epochs, sharp_hist['test_loss'], 'r--', lw=2, label='Sharp Test')
    ax2.plot(epochs, flat_hist['train_loss'], 'b-', lw=2, label='Flat Train')
    ax2.plot(epochs, flat_hist['test_loss'], 'b--', lw=2, label='Flat Test')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Train vs Test Loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Generalization gap
    ax3 = fig.add_subplot(3, 3, 3)
    sharp_gap = [t - v for t, v in zip(sharp_hist['train_acc'], sharp_hist['test_acc'])]
    flat_gap = [t - v for t, v in zip(flat_hist['train_acc'], flat_hist['test_acc'])]
    ax3.plot(epochs, sharp_gap, 'r-o', lw=2, label='Sharp')
    ax3.plot(epochs, flat_gap, 'b-s', lw=2, label='Flat')
    ax3.axhline(0, color='gray', ls='--', alpha=0.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train - Test Acc (%)")
    ax3.set_title("Generalization Gap")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Row 2: 2D Heatmaps
    for idx, (res, title, cmap) in enumerate([
        (sharp_res, "Sharp Model", "hot"),
        (flat_res, "Flat Model", "cool")
    ]):
        ax = fig.add_subplot(3, 3, idx + 4)
        r = res['range']
        im = ax.imshow(res['landscape_2d'], cmap=cmap, origin='lower', extent=[-r,r,-r,r])
        ax.plot(0, 0, 'w*', markersize=12)
        ax.set_title(f"{title}\nSharpness: {res['sharpness']:.4f}")
        ax.set_xlabel("Dir 1")
        ax.set_ylabel("Dir 2")
        plt.colorbar(im, ax=ax)
    
    # 1D Slice
    ax = fig.add_subplot(3, 3, 6)
    n, r = sharp_res['num_points'], sharp_res['range']
    x = np.linspace(-r, r, n)
    center = n // 2
    ax.plot(x, sharp_res['landscape_2d'][center], 'r-', lw=2, label='Sharp')
    ax.plot(x, flat_res['landscape_2d'][center], 'b-', lw=2, label='Flat')
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Loss")
    ax.set_title("1D Slice Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 3: 3D Surfaces + Summary
    for idx, (res, title, cmap) in enumerate([
        (sharp_res, "Sharp Model 3D", "hot"),
        (flat_res, "Flat Model 3D", "cool")
    ]):
        ax = fig.add_subplot(3, 3, idx + 7, projection='3d')
        n, r = res['num_points'], res['range']
        x = np.linspace(-r, r, n)
        X, Y = np.meshgrid(x, x)
        ax.plot_surface(X, Y, res['landscape_2d'], cmap=cmap, alpha=0.9, linewidth=0.1)
        ax.set_xlabel("Dir 1")
        ax.set_ylabel("Dir 2")
        ax.set_zlabel("Loss")
        ax.set_title(title)
    
    # Summary
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    summary = f"""
╔═════════════════════════════════════════╗
║            MNIST RESULTS                ║
╠═════════════════════════════════════════╣
║  Sharp Model (no regul, lr=0.05)        ║
║    Train Acc:  {sharp_hist['train_acc'][-1]:<6.1f}%               ║
║    Test Acc:   {sharp_hist['test_acc'][-1]:<6.1f}%               ║
║    Gap:        {sharp_hist['train_acc'][-1]-sharp_hist['test_acc'][-1]:<6.1f}%               ║
║    Sharpness:  {sharp_res['sharpness']:<8.4f}             ║
╠═════════════════════════════════════════╣
║  Flat Model (BN+Dropout, lr=0.01)       ║
║    Train Acc:  {flat_hist['train_acc'][-1]:<6.1f}%               ║
║    Test Acc:   {flat_hist['test_acc'][-1]:<6.1f}%               ║
║    Gap:        {flat_hist['train_acc'][-1]-flat_hist['test_acc'][-1]:<6.1f}%               ║
║    Sharpness:  {flat_res['sharpness']:<8.4f}             ║
╠═════════════════════════════════════════╣
║  Lower gap + lower sharpness = better   ║
╚═════════════════════════════════════════╝
    """
    ax.text(0.0, 0.95, summary, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('landscape_mnist_report.png', dpi=150)
    plt.close()
    print("✅ Saved: landscape_mnist_report.png")


# ============ MAIN ============

def main():
    print("=" * 50)
    print("🔍 Loss Landscape Analysis - MNIST")
    print("=" * 50)
    
    # Data
    print("\n📦 Loading MNIST...")
    train_loader, test_loader = load_mnist()
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Test:  {len(test_loader.dataset)} samples")
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Train Sharp Model
    print("\n🏋️ Sharp Model (lr=0.05, no regularization)")
    sharp_model = SharpModel()
    sharp_hist = train(sharp_model, train_loader, test_loader, lr=0.05, epochs=5)
    
    # Train Flat Model  
    print("\n🏋️ Flat Model (lr=0.01, BatchNorm + Dropout)")
    flat_model = FlatModel()
    flat_hist = train(flat_model, train_loader, test_loader, lr=0.01, epochs=5)
    
    # Analyze landscapes
    print("\n📊 Analyzing Sharp Model...")
    sharp_analyzer = LossLandscapeAnalyzer(sharp_model, loss_fn, train_loader)
    sharp_res = sharp_analyzer.analyze(num_points=21, range_val=1.0)

    print("\n📊 Analyzing Flat Model...")
    flat_analyzer = LossLandscapeAnalyzer(flat_model, loss_fn, train_loader)
    flat_res = flat_analyzer.analyze(num_points=21, range_val=1.0)

    # Plot
    plot_results(sharp_res, flat_res, sharp_hist, flat_hist)
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FINAL COMPARISON")
    print("=" * 50)
    print(f"""
    Model        | Test Acc | Gap    | Sharpness
    -------------|----------|--------|----------
    Sharp        | {sharp_hist['test_acc'][-1]:.1f}%   | {sharp_hist['train_acc'][-1]-sharp_hist['test_acc'][-1]:.1f}%  | {sharp_res['sharpness']:.4f}
    Flat         | {flat_hist['test_acc'][-1]:.1f}%   | {flat_hist['train_acc'][-1]-flat_hist['test_acc'][-1]:.1f}%  | {flat_res['sharpness']:.4f}
    """)
    
    print("✅ Done!")


if __name__ == "__main__":
    main()