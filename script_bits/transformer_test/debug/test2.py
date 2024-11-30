import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple regression model
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simulate a regression dataset (input size 10, output size 1)
def generate_data(n_samples=10000, input_size=10):
    X = torch.randn(n_samples, input_size)
    y = torch.sum(X, dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1  # y = sum(X) + noise
    return X, y

# Generate data
X, y = generate_data()

# Create DataLoader with large batch size
batch_size = 2048
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = 10
output_size = 1
model = RegressionModel(input_size, output_size)

# Use DataParallel to distribute the model across multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
