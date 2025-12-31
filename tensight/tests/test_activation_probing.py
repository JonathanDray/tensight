import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from tensight.analyzers.activation_probing import ActivationProber


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 512)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(128, 64)
        self.act4 = nn.ReLU()
        self.layer5 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.layer5(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x


def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def train_model(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        
        acc = 100 * correct / total
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f} Acc={acc:.1f}%")
    
    return acc


def plot_probing_results(results_dict, filename):
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 5))
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        layer_results = results['layer_results']
        layers = list(layer_results.keys())
        accs = [layer_results[l]['test_accuracy'] * 100 for l in layers]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
        bars = ax.barh(range(len(layers)), accs, color=colors)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=9)
        ax.set_xlabel('Probe Accuracy (%)')
        ax.set_title(f'{model_name}\nProbing Results')
        ax.set_xlim(0, 105)
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Random (10%)')
        ax.legend(fontsize=8)
        
        for bar, acc in zip(bars, accs):
            ax.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    ax = axes[-1]
    ax.axis('off')
    
    summary = """
╔═══════════════════════════════════════════╗
║       ACTIVATION PROBING SUMMARY          ║
╠═══════════════════════════════════════════╣
║                                           ║
║  • Early layers: low accuracy             ║
║    → Raw features, no class info yet      ║
║                                           ║
║  • Middle layers: rising accuracy         ║
║    → Class info being formed              ║
║                                           ║
║  • Late layers: high accuracy             ║
║    → Class fully encoded                  ║
║                                           ║
╠═══════════════════════════════════════════╣
║                                           ║
║  Use cases:                               ║
║  • Find where features emerge             ║
║  • Compare architectures                  ║
║  • Debug representation learning          ║
║  • Choose layers for transfer learning    ║
║                                           ║
╚═══════════════════════════════════════════╝
    """
    
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")


def main():
    print("=" * 50)
    print("🔬 Activation Probing Analysis - MNIST")
    print("=" * 50)
    
    print("\n📦 Loading MNIST...")
    train_loader, test_loader = load_mnist()
    
    results_dict = {}
    
    print("\n" + "-" * 50)
    print("🧠 Model 1: Deep MLP")
    print("-" * 50)
    
    mlp = DeepMLP()
    print("\n🏋️ Training...")
    train_model(mlp, train_loader, epochs=3)
    
    print("\n🔬 Probing layers...")
    prober_mlp = ActivationProber(mlp)
    results_mlp = prober_mlp.probe(
        train_loader=train_loader,
        test_loader=test_loader,
        layer_names=['layer1', 'layer2', 'layer3', 'layer4', 'layer5'],
        max_samples=2000
    )
    results_dict['Deep MLP'] = results_mlp
    
    print("\n" + "-" * 50)
    print("🧠 Model 2: Simple CNN")
    print("-" * 50)
    
    cnn = SimpleCNN()
    print("\n🏋️ Training...")
    train_model(cnn, train_loader, epochs=3)
    
    print("\n🔬 Probing layers...")
    prober_cnn = ActivationProber(cnn)
    results_cnn = prober_cnn.probe(
        train_loader=train_loader,
        test_loader=test_loader,
        layer_names=['conv1', 'conv2', 'fc1', 'fc2'],
        max_samples=2000
    )
    results_dict['Simple CNN'] = results_cnn
    
    print("\n" + "-" * 50)
    print("📈 Generating visualization...")
    plot_probing_results(results_dict, 'probing_report.png')
    
    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    
    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        layer_results = results['layer_results']
        
        for layer, res in layer_results.items():
            acc = res['test_accuracy'] * 100
            interpretation = ""
            if acc < 30:
                interpretation = "(raw features)"
            elif acc < 70:
                interpretation = "(forming)"
            else:
                interpretation = "(class encoded)"
            print(f"   {layer:<15} {acc:>6.1f}%  {interpretation}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()