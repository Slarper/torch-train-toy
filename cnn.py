import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Fully Connected Layer (fully connected after flattening the output of the conv layers)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming input image size is 32x32
        self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes for classification

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Pass through the first convolutional layer and apply ReLU activation
        x = F.relu(self.conv1(x))
        # Max Pooling
        x = F.max_pool2d(x, 2)

        # Pass through the second convolutional layer and apply ReLU activation
        x = F.relu(self.conv2(x))
        # Max Pooling
        x = F.max_pool2d(x, 2)

        # Pass through the third convolutional layer and apply ReLU activation
        x = F.relu(self.conv3(x))
        # Max Pooling
        x = F.max_pool2d(x, 2)

        # Flatten the output from 3D tensor to 1D vector
        x = x.view(-1, 128 * 8 * 8)

        # Pass through the first fully connected layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)
        return x
