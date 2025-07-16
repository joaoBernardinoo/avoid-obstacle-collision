import torch
import torch.nn as nn
from torchinfo import summary

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

if __name__ == "__main__":
    # Create an instance of the model
    model = BallAngleCNN(input_channels=1).to("cuda")

    # Define the input size (batch_size, channels, height, width)
    input_size = (1, 1, 40, 200) 

    # Print the summary
    summary(model, input_size=input_size)
    # Assuming 'model' is your instantiated BallAngleCNN
    # and 'input_size' is defined as before

    # Create a dummy input tensor with the correct size
    dummy_input = torch.randn(input_size)

    # Move model and dummy_input to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Export the model
    torch.onnx.export(model,
                    (dummy_input,),
                    "ball_angle_cnn.onnx", # where to save the file
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output'])
