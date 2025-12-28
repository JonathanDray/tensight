import torch
import torch.nn as nn
import torch.optim as optim
import tensight



class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.Sigmoid()  
        
        
        nn.init.normal_(self.fc1.weight, std=5.0)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)



X = torch.randn(100, 10) * 1000  
y = torch.zeros(100, 1)
y[:10] = 1  


config = {'lr': 0.5, 'batch_size': 4, 'epochs': 100}


print("=" * 60)
print("TESTING PRE-CHECK")
print("=" * 60)

model = BadModel()
report = tensight.pre_check(model, (X, y), config)


print("\n" + "=" * 60)
print("TESTING WATCHER")
print("=" * 60)

model = BadModel()
model = tensight.watch(model)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    pred = model(X)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.record_loss(loss)

report = model.diagnose()

