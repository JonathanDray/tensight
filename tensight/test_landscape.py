import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


import tensight
from tensight.analyzers import LossLandscapeAnalyzer






print("📦 Loading MNIST...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)




train_loader_sharp = DataLoader(train_dataset, batch_size=16, shuffle=True)   
train_loader_flat = DataLoader(train_dataset, batch_size=512, shuffle=True)   
train_loader_eval = DataLoader(train_dataset, batch_size=256, shuffle=False)  

print(f"   Loaded {len(train_dataset)} samples")






class SharpModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.net(x)


class FlatModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)






def train_sharp(model, epochs=8):
    
    print("\n🔴 Training SHARP model...")
    print("   Config: lr=0.1, batch=16, no regularization, SGD")
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)  
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader_sharp):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            
            if batch_idx >= 500:
                break
        
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
    
    return model


def train_flat(model, epochs=8):
    
    print("\n🔵 Training FLAT model...")
    print("   Config: lr=0.01, batch=512, weight_decay=0.01, momentum=0.9")
    
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01,           
        momentum=0.9,      
        weight_decay=0.01  
    )
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader_flat):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx >= 100:  
                break
        
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
    
    return model






sharp_model = SharpModel()
flat_model = FlatModel()

sharp_model = train_sharp(sharp_model, epochs=10)
flat_model = train_flat(flat_model, epochs=10)






criterion = nn.CrossEntropyLoss()

print("\n" + "=" * 60)
print("🗺️ LOSS LANDSCAPE ANALYSIS")
print("=" * 60)


RANGE_VAL = 0.5      
NUM_POINTS = 31      
NUM_BATCHES = 10     

print("\n▶ Analyzing Sharp Model...")
sharp_analyzer = LossLandscapeAnalyzer(
    model=sharp_model,
    loss_fn=criterion,
    data_loader=train_loader_eval
)
sharp_results = sharp_analyzer.analyze(
    num_points=NUM_POINTS, 
    range_val=RANGE_VAL,
    num_batches=NUM_BATCHES
)

print("\n▶ Analyzing Flat Model...")
flat_analyzer = LossLandscapeAnalyzer(
    model=flat_model,
    loss_fn=criterion,
    data_loader=train_loader_eval
)
flat_results = flat_analyzer.analyze(
    num_points=NUM_POINTS, 
    range_val=RANGE_VAL,
    num_batches=NUM_BATCHES
)






def compute_extra_metrics(landscape, center_loss):
    
    
    
    variance = np.var(landscape)
    
    
    ratio = np.max(landscape) / (np.min(landscape) + 1e-10)
    
    
    grad_x = np.abs(np.diff(landscape, axis=0)).mean()
    grad_y = np.abs(np.diff(landscape, axis=1)).mean()
    avg_gradient = (grad_x + grad_y) / 2
    
    
    relative_sharpness = (np.max(landscape) - center_loss) / (center_loss + 1e-10)
    
    return {
        'variance': variance,
        'max_min_ratio': ratio,
        'avg_gradient': avg_gradient,
        'relative_sharpness': relative_sharpness
    }

sharp_extra = compute_extra_metrics(sharp_results['landscape_2d'], sharp_results['center_loss'])
flat_extra = compute_extra_metrics(flat_results['landscape_2d'], flat_results['center_loss'])






print("\n" + "=" * 60)
print("📊 DETAILED COMPARISON")
print("=" * 60)

┌────────────────────────────────────────────────────────────────┐
│                    SHARPNESS METRICS                           │
├────────────────────────────────────────────────────────────────┤
│  Metric               │  Sharp Model    │  Flat Model          │
├────────────────────────────────────────────────────────────────┤
│  Center Loss          │  {sharp_results['center_loss']:>12.4f}  │  {flat_results['center_loss']:>12.4f}       │
│  Avg Sharpness        │  {sharp_results['sharpness']:>12.4f}  │  {flat_results['sharpness']:>12.4f}       │
│  Max Sharpness        │  {sharp_results['sharpness_max']:>12.4f}  │  {flat_results['sharpness_max']:>12.4f}       │
│  Landscape Variance   │  {sharp_extra['variance']:>12.4f}  │  {flat_extra['variance']:>12.4f}       │
│  Max/Min Ratio        │  {sharp_extra['max_min_ratio']:>12.2f}  │  {flat_extra['max_min_ratio']:>12.2f}       │
│  Avg Gradient         │  {sharp_extra['avg_gradient']:>12.4f}  │  {flat_extra['avg_gradient']:>12.4f}       │
│  Relative Sharpness   │  {sharp_extra['relative_sharpness']:>12.2f}%  │  {flat_extra['relative_sharpness']:>12.2f}%       │
└────────────────────────────────────────────────────────────────┘






print("\n📈 Generating visualizations...")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


vmin = min(sharp_results['landscape_2d'].min(), flat_results['landscape_2d'].min())
vmax = max(sharp_results['landscape_2d'].max(), flat_results['landscape_2d'].max())

im1 = axes[0].imshow(
    sharp_results['landscape_2d'], 
    cmap='hot',
    origin='lower',
    extent=[-RANGE_VAL, RANGE_VAL, -RANGE_VAL, RANGE_VAL],
    vmin=vmin, vmax=vmax
)
axes[0].set_title(f"Sharp Model\nSharpness={sharp_results['sharpness']:.4f}, Var={sharp_extra['variance']:.4f}", fontsize=11)
axes[0].set_xlabel("Direction 1")
axes[0].set_ylabel("Direction 2")
axes[0].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im1, ax=axes[0], label='Loss')

im2 = axes[1].imshow(
    flat_results['landscape_2d'],
    cmap='cool',
    origin='lower',
    extent=[-RANGE_VAL, RANGE_VAL, -RANGE_VAL, RANGE_VAL],
    vmin=vmin, vmax=vmax
)
axes[1].set_title(f"Flat Model\nSharpness={flat_results['sharpness']:.4f}, Var={flat_extra['variance']:.4f}", fontsize=11)
axes[1].set_xlabel("Direction 1")
axes[1].set_ylabel("Direction 2")
axes[1].plot(0, 0, 'w*', markersize=15)
plt.colorbar(im2, ax=axes[1], label='Loss')

plt.suptitle("Loss Landscape Comparison - Same Color Scale", fontsize=13)
plt.tight_layout()
plt.savefig('landscape_comparison_v2.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: landscape_comparison_v2.png")



fig = plt.figure(figsize=(16, 6))

X, Y = np.meshgrid(
    np.linspace(-RANGE_VAL, RANGE_VAL, NUM_POINTS),
    np.linspace(-RANGE_VAL, RANGE_VAL, NUM_POINTS)
)

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, sharp_results['landscape_2d'], cmap='hot', alpha=0.9, edgecolor='none')
ax1.set_title(f"Sharp Model\nVariance: {sharp_extra['variance']:.4f}", fontsize=12)
ax1.set_xlabel("Dir 1")
ax1.set_ylabel("Dir 2")
ax1.set_zlabel("Loss")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, flat_results['landscape_2d'], cmap='cool', alpha=0.9, edgecolor='none')
ax2.set_title(f"Flat Model\nVariance: {flat_extra['variance']:.4f}", fontsize=12)
ax2.set_xlabel("Dir 1")
ax2.set_ylabel("Dir 2")
ax2.set_zlabel("Loss")


z_max = max(ax1.get_zlim()[1], ax2.get_zlim()[1])
ax1.set_zlim(0, z_max)
ax2.set_zlim(0, z_max)

plt.tight_layout()
plt.savefig('landscape_3d_v2.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: landscape_3d_v2.png")



fig, ax = plt.subplots(figsize=(10, 6))

center = NUM_POINTS // 2
x_range = np.linspace(-RANGE_VAL, RANGE_VAL, NUM_POINTS)

sharp_slice = sharp_results['landscape_2d'][center, :]
flat_slice = flat_results['landscape_2d'][center, :]

ax.plot(x_range, sharp_slice, 'r-', linewidth=2.5, label=f'Sharp (var={sharp_extra["variance"]:.4f})')
ax.plot(x_range, flat_slice, 'b-', linewidth=2.5, label=f'Flat (var={flat_extra["variance"]:.4f})')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(x_range, sharp_slice, alpha=0.3, color='red')
ax.fill_between(x_range, flat_slice, alpha=0.3, color='blue')

ax.set_xlabel("Perturbation (Direction 1)", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("1D Slice Through Center - Sharp vs Flat", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('landscape_1d_overlay.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: landscape_1d_overlay.png")






print("\n" + "=" * 60)
print("🧪 GENERALIZATION TEST")
print("=" * 60)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

def evaluate(model, name):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    return acc, avg_loss

sharp_acc, sharp_test_loss = evaluate(sharp_model, "Sharp")
flat_acc, flat_test_loss = evaluate(flat_model, "Flat")

┌────────────────────────────────────────────────────────────────┐
│                    GENERALIZATION RESULTS                      │
├────────────────────────────────────────────────────────────────┤
│  Model        │  Train Loss  │  Test Loss   │  Test Accuracy   │
├────────────────────────────────────────────────────────────────┤
│  Sharp Model  │  {sharp_results['center_loss']:>10.4f}  │  {sharp_test_loss:>10.4f}  │  {sharp_acc:>10.2f}%      │
│  Flat Model   │  {flat_results['center_loss']:>10.4f}  │  {flat_test_loss:>10.4f}  │  {flat_acc:>10.2f}%      │
└────────────────────────────────────────────────────────────────┘

📚 Analysis:
   • Gap (Test-Train Loss): Sharp={sharp_test_loss - sharp_results['center_loss']:.4f}, Flat={flat_test_loss - flat_results['center_loss']:.4f}
   • Lower gap = better generalization


if flat_extra['variance'] < sharp_extra['variance']:
    print("✅ Flat model has lower landscape variance (flatter minimum)")
if flat_acc >= sharp_acc - 1:  
    print("✅ Flat model generalizes as well or better")






print("\n" + "=" * 60)
print("🏆 SUMMARY")
print("=" * 60)

📊 Key Findings:

   Sharp Model (high LR, small batch, no regularization):
   • Landscape variance: {sharp_extra['variance']:.4f}
   • Max sharpness: {sharp_results['sharpness_max']:.4f}
   • Test accuracy: {sharp_acc:.2f}%

   Flat Model (low LR, large batch, weight decay + dropout + batchnorm):
   • Landscape variance: {flat_extra['variance']:.4f}
   • Max sharpness: {flat_results['sharpness_max']:.4f}
   • Test accuracy: {flat_acc:.2f}%

   Variance ratio (Sharp/Flat): {sharp_extra['variance'] / (flat_extra['variance'] + 1e-10):.2f}x

📁 Generated files:
   • landscape_comparison_v2.png
   • landscape_3d_v2.png
   • landscape_1d_overlay.png

🔍 Tensight - See through your models