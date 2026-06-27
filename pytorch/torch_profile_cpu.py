import torch
import torch.nn as nn
import torch.optim as optim
import time

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
        x = self.fc3(x)
        return x

# ---------- Synthetic training data ----------
torch.manual_seed(42)
n_samples = 100
xs = torch.rand(n_samples, 3) * 4.0 - 2.0
ys = 2.0 * xs[:, 0] - 3.0 * xs[:, 1] + 1.5 * xs[:, 2] + 0.5
ys = ys.view(-1, 1)

device = torch.device('cpu')
model = MLP().to(device)
xs = xs.to(device)
ys = ys.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 200
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()

    # Accumulate squared errors over the whole batch
    total_sq_err = torch.tensor(0.0, device=device)

    # Loop over each sample individually (like Rust)
    for i in range(n_samples):
        x_i = xs[i:i+1]          # shape (1, 3)
        y_i = ys[i:i+1]          # shape (1, 1)

        pred_i = model(x_i)
        se = ((pred_i - y_i) ** 2).squeeze()   # now a scalar
        total_sq_err += se

    loss = total_sq_err / n_samples

    loss.backward()

    # Manual parameter update (exactly like Rust)
    with torch.no_grad():
        for param in model.parameters():
            param.data -= 0.01 * param.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3}: loss = {loss.item():.6f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.4f} seconds")

# ---------- Test ----------
test_x = torch.tensor([[1.5, -0.5, 0.7]], device=device)
with torch.no_grad():
    test_out = model(test_x)
expected = 2.0 * 1.5 - 3.0 * (-0.5) + 1.5 * 0.7 + 0.5
print(f"\nAfter training:\n  f(1.5, -0.5, 0.7) = {test_out.item():.6f}  (expected {expected:.6f})")