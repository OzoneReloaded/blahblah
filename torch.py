import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet(input_dim=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    logits = model(X)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")