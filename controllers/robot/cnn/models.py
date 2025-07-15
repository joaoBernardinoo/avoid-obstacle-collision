import torch
import torch.nn as nn


class BallAngleCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(BallAngleCNN, self).__init__()
        # Flexible input channels (works for 1 or 3 channels)
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size automatically
        self.flatten = nn.Flatten()
        # Adjust these numbers if needed
        self.fc1 = nn.Linear(64 * 10 * 50, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()
