import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate training data: y = x^2
x_train = np.linspace(-10, 10, 100)  # 100 points from -10 to 10
y_train = x_train**7  # y = x^2

y_max = y_train.max()  # to normalize y values to [0, 10]
y_train /= y_max
y_train *= 10
# Plot the loss curve
# plt.scatter(
#     x_train,
#     y_train,
# )
# plt.title("Training Dataset")
# plt.xlabel("Output")
# plt.ylabel("Input")
# plt.show()

# Convert numpy arrays to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # A simple network with one hidden layer
        self.fc1 = nn.Linear(1, 64)  # 1 input, 64 neurons in the hidden layer
        self.fc11 = nn.Linear(64, 128)
        self.fc12 = nn.Linear(128, 128)
        self.fc13 = nn.Linear(128, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # 64 neurons to 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = self.fc2(x)
        return x


# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
losses = []

for epoch in range(epochs):
    model.train()

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # Backward pass: Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Plot the loss curve
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Plot the results
model.eval()
with torch.no_grad():
    y_pred = model(x_train)

plt.scatter(x_train.numpy(), y_train.numpy(), label="True values", color="blue")
plt.plot(x_train.numpy(), y_pred.numpy(), label="Predicted values", color="red")
plt.title("True vs Predicted (y = x^3)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
