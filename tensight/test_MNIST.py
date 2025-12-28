import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from tensight.analyzers.loss_landscape import LossLandscapeAnalyzer






print("📦 Loading MNIST...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False,  
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"   Train: {len(train_dataset)} samples")
print(f"   Test:  {len(test_dataset)} samples")






class SharpModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class FlatModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.act(self.bn1(self.fc1(x))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))
        return self.fc3(x)






def train_model(model, name, lr, epochs=5, max_batches=300):
    print(f"\n🏋️ Training {name}...")
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx >= max_batches:
                break
        
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
    
    return model



print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

sharp_model = SharpModel()
sharp_model = train_model(sharp_model, "Sharp Model (high LR, no reg)", lr=0.1, epochs=5)

flat_model = FlatModel()
flat_model = train_model(flat_model, "Flat Model (low LR, with reg)", lr=0.01, epochs=5)






def evaluate(model, name):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    acc = 100. * correct / total
    print(f"   {name}: {acc:.2f}%")
    return acc

print("\n📊 Test Accuracy:")
sharp_acc = evaluate(sharp_model, "Sharp Model")
flat_acc = evaluate(flat_model, "Flat Model")






criterion = nn.CrossEntropyLoss()

print("\n" + "=" * 60)
print("🗺️ LOSS LANDSCAPE: WITHOUT Filter Normalization")
print("=" * 60)

print("\n--- Sharp Model ---")
analyzer_sharp = LossLandscapeAnalyzer(sharp_model, criterion, train_loader)
results_sharp_no_norm = analyzer_sharp.analyze(
    num_points=21, 
    range_val=1.0, 
    use_filter_norm=False
)

print("\n--- Flat Model ---")
analyzer_flat = LossLandscapeAnalyzer(flat_model, criterion, train_loader)
results_flat_no_norm = analyzer_flat.analyze(
    num_points=21, 
    range_val=1.0, 
    use_filter_norm=False
)


print("\n" + "=" * 60)
print("🗺️ LOSS LANDSCAPE: WITH Filter Normalization")
print("=" * 60)

print("\n--- Sharp Model ---")
results_sharp_norm = analyzer_sharp.analyze(
    num_points=21, 
    range_val=1.0, 
    use_filter_norm=True
)

print("\n--- Flat Model ---")
results_flat_norm = analyzer_flat.analyze(
    num_points=21, 
    range_val=1.0, 
    use_filter_norm=True
)






print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY")
print("=" * 70)

┌──────────────────────────────────────────────────────────────────────┐
│                    FILTER NORMALIZATION COMPARISON                   │
├──────────────────────────────────────────────────────────────────────┤
│                    │   Without Filter Norm   │   With Filter Norm    │
├──────────────────────────────────────────────────────────────────────┤
│  Sharp Model       │                         │                       │
│    - Sharpness     │   {results_sharp_no_norm['sharpness']:<20.4f}  │   {results_sharp_norm['sharpness']:<20.4f}│
│    - Max Sharpness │   {results_sharp_no_norm['sharpness_max']:<20.4f}  │   {results_sharp_norm['sharpness_max']:<20.4f}│
├──────────────────────────────────────────────────────────────────────┤
│  Flat Model        │                         │                       │
│    - Sharpness     │   {results_flat_no_norm['sharpness']:<20.4f}  │   {results_flat_norm['sharpness']:<20.4f}│
│    - Max Sharpness │   {results_flat_no_norm['sharpness_max']:<20.4f}  │   {results_flat_norm['sharpness_max']:<20.4f}│
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    GENERALIZATION COMPARISON                         │
├──────────────────────────────────────────────────────────────────────┤
│  Model        │  Sharpness (norm)  │  Test Accuracy  │  Correlation │
├──────────────────────────────────────────────────────────────────────┤
│  Sharp Model  │  {results_sharp_norm['sharpness']:<18.4f}│  {sharp_acc:<15.2f}%│               │
│  Flat Model   │  {results_flat_norm['sharpness']:<18.4f}│  {flat_acc:<15.2f}%│               │
└──────────────────────────────────────────────────────────────────────┘


if results_flat_norm['sharpness'] < results_sharp_norm['sharpness'] and flat_acc > sharp_acc:
    print("✅ Theory confirmed: Flatter minimum → Better generalization!")
elif results_flat_norm['sharpness'] > results_sharp_norm['sharpness'] and flat_acc < sharp_acc:
    print("✅ Theory confirmed: Sharper minimum → Worse generalization!")
else:
    print("⚠️ Results don't match theory (this can happen with limited training)")






print("\n📈 Generating visualizations...")


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

im1 = axes[0, 0].imshow(
    results_sharp_no_norm['landscape_2d'], 
    cmap='hot', origin='lower', extent=[-1,1,-1,1]
)
axes[0, 0].set_title(f"Sharp Model - NO Filter Norm\nSharpness: {results_sharp_no_norm['sharpness']:.4f}")
axes[0, 0].set_xlabel("Direction 1")
axes[0, 0].set_ylabel("Direction 2")
axes[0, 0].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im1, ax=axes[0, 0], label='Loss')

im2 = axes[0, 1].imshow(
    results_flat_no_norm['landscape_2d'], 
    cmap='cool', origin='lower', extent=[-1,1,-1,1]
)
axes[0, 1].set_title(f"Flat Model - NO Filter Norm\nSharpness: {results_flat_no_norm['sharpness']:.4f}")
axes[0, 1].set_xlabel("Direction 1")
axes[0, 1].set_ylabel("Direction 2")
axes[0, 1].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im2, ax=axes[0, 1], label='Loss')

im3 = axes[1, 0].imshow(
    results_sharp_norm['landscape_2d'], 
    cmap='hot', origin='lower', extent=[-1,1,-1,1]
)
axes[1, 0].set_title(f"Sharp Model - WITH Filter Norm\nSharpness: {results_sharp_norm['sharpness']:.4f}")
axes[1, 0].set_xlabel("Direction 1")
axes[1, 0].set_ylabel("Direction 2")
axes[1, 0].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im3, ax=axes[1, 0], label='Loss')

im4 = axes[1, 1].imshow(
    results_flat_norm['landscape_2d'], 
    cmap='cool', origin='lower', extent=[-1,1,-1,1]
)
axes[1, 1].set_title(f"Flat Model - WITH Filter Norm\nSharpness: {results_flat_norm['sharpness']:.4f}")
axes[1, 1].set_xlabel("Direction 1")
axes[1, 1].set_ylabel("Direction 2")
axes[1, 1].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im4, ax=axes[1, 1], label='Loss')

plt.suptitle("Effect of Filter Normalization on Loss Landscape Visualization\n(Li et al., NeurIPS 2018)", fontsize=14)
plt.tight_layout()
plt.savefig('mnist_filter_norm_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: mnist_filter_norm_comparison.png")



fig = plt.figure(figsize=(16, 12))

x = np.linspace(-1, 1, 21)
X, Y = np.meshgrid(x, x)

ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, Y, results_sharp_no_norm['landscape_2d'], cmap='hot', alpha=0.8)
ax1.set_title(f"Sharp - No Norm\n(sharpness={results_sharp_no_norm['sharpness']:.4f})")
ax1.set_xlabel("Dir 1")
ax1.set_ylabel("Dir 2")
ax1.set_zlabel("Loss")

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, Y, results_flat_no_norm['landscape_2d'], cmap='cool', alpha=0.8)
ax2.set_title(f"Flat - No Norm\n(sharpness={results_flat_no_norm['sharpness']:.4f})")
ax2.set_xlabel("Dir 1")
ax2.set_ylabel("Dir 2")
ax2.set_zlabel("Loss")

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(X, Y, results_sharp_norm['landscape_2d'], cmap='hot', alpha=0.8)
ax3.set_title(f"Sharp - Filter Norm\n(sharpness={results_sharp_norm['sharpness']:.4f})")
ax3.set_xlabel("Dir 1")
ax3.set_ylabel("Dir 2")
ax3.set_zlabel("Loss")

ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(X, Y, results_flat_norm['landscape_2d'], cmap='cool', alpha=0.8)
ax4.set_title(f"Flat - Filter Norm\n(sharpness={results_flat_norm['sharpness']:.4f})")
ax4.set_xlabel("Dir 1")
ax4.set_ylabel("Dir 2")
ax4.set_zlabel("Loss")

plt.suptitle("3D Loss Landscapes - MNIST", fontsize=14)
plt.tight_layout()
plt.savefig('mnist_landscape_3d.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: mnist_landscape_3d.png")



fig, axes = plt.subplots(1, 2, figsize=(14, 5))

center = 10  
x_range = np.linspace(-1, 1, 21)


axes[0].plot(x_range, results_sharp_norm['landscape_2d'][center, :], 'r-', linewidth=2, label='Sharp Model')
axes[0].plot(x_range, results_flat_norm['landscape_2d'][center, :], 'b-', linewidth=2, label='Flat Model')
axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel("Direction 1")
axes[0].set_ylabel("Loss")
axes[0].set_title("1D Slice - WITH Filter Normalization")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


axes[1].plot(x_range, results_sharp_no_norm['landscape_2d'][center, :], 'r-', linewidth=2, label='Sharp Model')
axes[1].plot(x_range, results_flat_no_norm['landscape_2d'][center, :], 'b-', linewidth=2, label='Flat Model')
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Direction 1")
axes[1].set_ylabel("Loss")
axes[1].set_title("1D Slice - WITHOUT Filter Normalization")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("1D Loss Landscape Slices - MNIST", fontsize=14)
plt.tight_layout()
plt.savefig('mnist_landscape_1d.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: mnist_landscape_1d.png")






print("\n" + "=" * 70)
print("📚 EXPLANATION: Filter Normalization")
print("=" * 70)

🔬 What is Filter Normalization?

   From Li et al., NeurIPS 2018:
   
   "The sharpness of a minimum is sensitive to the scale of the weights. 
   To make fair comparisons, we normalize each filter in the random 
   direction d to have the same norm as the corresponding filter in θ."
   
   Formula for each filter i:
   
       d_i = d_i * ||θ_i|| / ||d_i||
   
   Where:
   - θ_i = original weights of filter i  
   - d_i = random direction for filter i

🎯 Why it matters:

   WITHOUT normalization:
   ┌────────────────────────────────────────┐
   │ Layer 1 weights: ||w|| = 10.0         │
   │ Layer 2 weights: ||w|| = 0.1          │
   │                                        │
   │ Random direction: ||d|| = 1.0         │
   │                                        │
   │ → Layer 1 perturbed by 10%            │
   │ → Layer 2 perturbed by 1000%!         │
   │                                        │
   │ Result: Unfair, dominated by Layer 2  │
   └────────────────────────────────────────┘
   
   WITH normalization:
   ┌────────────────────────────────────────┐
   │ Layer 1: d normalized to ||d|| = 10.0 │
   │ Layer 2: d normalized to ||d|| = 0.1  │
   │                                        │
   │ → Layer 1 perturbed by ~100%          │
   │ → Layer 2 perturbed by ~100%          │
   │                                        │
   │ Result: Fair comparison across layers │
   └────────────────────────────────────────┘

📄 Reference:
   Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).
   "Visualizing the Loss Landscape of Neural Nets"
   NeurIPS 2018


print("\n🏆 DONE!")
   Generated files:
   • mnist_filter_norm_comparison.png
   • mnist_landscape_3d.png  
   • mnist_landscape_1d.png
   
   Key findings:
   • Sharp Model sharpness (norm): {results_sharp_norm['sharpness']:.4f}
   • Flat Model sharpness (norm):  {results_flat_norm['sharpness']:.4f}
   • Sharp Model test acc: {sharp_acc:.2f}%
   • Flat Model test acc:  {flat_acc:.2f}%

🔍 Tensight - See through your models