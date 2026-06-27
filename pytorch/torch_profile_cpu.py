import torch
import torch.nn as nn
import torch.optim as optim
import time  # <-- added for profiling

# ---------- Network definition (3 -> 4 -> 4 -> 1) ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)          # no activation on last layer (linear)
        return x

# ---------- Synthetic training data ----------
torch.manual_seed(42)
n_samples = 100
xs = torch.rand(n_samples, 3) * 4.0 - 2.0   # uniform [-2, 2]
# Target: 2*x1 - 3*x2 + 1.5*x3 + 0.5
ys = 2.0 * xs[:, 0] - 3.0 * xs[:, 1] + 1.5 * xs[:, 2] + 0.5
ys = ys.view(-1, 1)   # shape (100, 1)

# ---------- Device (explicitly CPU, but can be changed) ----------
device = torch.device('cpu')   # or 'cuda' if available
model = MLP().to(device)
xs = xs.to(device)
ys = ys.to(device)

# ---------- Model, loss, optimizer ----------
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---------- Training loop with timing ----------
epochs = 200
start_time = time.time()   # start profiling

for epoch in range(epochs):
    # Forward pass
    pred = model(xs)
    loss = criterion(pred, ys)

    # Backward pass & gradient update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3}: loss = {loss.item():.6f}")

end_time = time.time()     # end profiling
elapsed = end_time - start_time
print(f"\nTraining completed in {elapsed:.4f} seconds")

# ---------- Test on a new input ----------
test_x = torch.tensor([[1.5, -0.5, 0.7]], device=device)
with torch.no_grad():
    test_out = model(test_x)
expected = 2.0 * 1.5 - 3.0 * (-0.5) + 1.5 * 0.7 + 0.5
print(f"\nAfter training:\n  f(1.5, -0.5, 0.7) = {test_out.item():.6f}  (expected {expected:.6f})")