import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Check if CUDA is available, and use GPU if possible, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate training data: y = x^2
x_train = np.linspace(-10, 10, 100)  # 100 points from -10 to 10
y_train = x_train**2  # y = x^2

y_max = y_train.max()  # to normalize y values to [0, 1]
y_train /= y_max

# Convert numpy arrays to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)  # 100x1 tensor
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 100x1 tensor

# Reshape the input to be "1-channel" images (100 samples, 1 channel, 1x1 size)
x_train = x_train.view(-1, 1, 1, 1)  # Reshape to (100, 1, 1, 1)

x_train, y_train = x_train.to(device), y_train.to(device)


# Define the Convolutional Neural Network (CNN) model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First layer: Convolution to map the input to a larger matrix (expand spatial dimensions)

        self.fc2 = nn.Linear(1, 8 * 8)

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # 1 input channel -> 16 output channels
        # Second layer: Convolution layer
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # 16 input channels -> 32 output channels
        # Third layer: Convolution layer
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # 32 input channels -> 64 output channels
        # Fourth layer: Convolution layer
        self.conv4 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # 64 input channels -> 128 output channels
        # Fully connected layer to output the final prediction
        self.fc1 = nn.Linear(8 * 8 * 128, 1)  # 128 channels to 1 output

    def forward(self, x):
        # Apply the four convolutional layers with ReLU activations
        x = self.fc2(x)
        x = x.view(-1, 1, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the output of the last convolution layer

        # Apply the fully connected layer
        x = self.fc1(x)
        return x


# Initialize the model, loss function, and optimizer
model = CNNModel().to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
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
# plt.plot(losses)
# plt.title("Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig('cnn_square_loss.png')

# Plot the results
model.eval()

x_pred = np.linspace(-10, 10, 1000)  # including more samples not in the training data

# Convert numpy arrays to torch tensors
x_pred = torch.tensor(x_pred, dtype=torch.float32).view(-1, 1)
x_pred = x_pred.to(device)
with torch.no_grad():
    y_pred = model(x_pred)

plt.scatter(
    x_train.cpu().numpy().squeeze(),
    y_train.cpu().numpy().squeeze(),
    label="True values",
    color="blue",
)
plt.plot(
    x_pred.cpu().numpy().squeeze(),
    y_pred.cpu().numpy().squeeze(),
    label="Predicted values",
    color="red",
)

# plt.figure()
plt.title("True vs Predicted (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('cnn_square.png')
