# Import necessary libraries
import numpy as np
import torch
from pathlib import Path
import sys
import torch.nn as nn
import torch.optim as optim
from .cnn_model import CNNNavigationModel

# Ensure the cnn_model module can be found
# This assumes the script is run from a location where this relative path is valid.



SCRIPT_DIR = Path(__file__).parent if '__file__' in locals() else Path.cwd()
MODEL_PATH = SCRIPT_DIR.parent / 'cnn' / 'cnn_navigation_model.pth'
ONLINE_MODEL_SAVE_PATH = SCRIPT_DIR / 'checkpoints' / 'online_cnn_model.pth'

# Ensure the checkpoints directory exists
ONLINE_MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_model():
    """Loads the pre-trained CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set lidar data shape to match the pre-trained model's expected input size
    model = CNNNavigationModel(lidar_shape_in=20).to(device)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded pre-trained model from {MODEL_PATH}")
    else:
        print(
            f"Warning: Model file not found at {MODEL_PATH}. Using an untrained model.")

    model.eval()
    return model, device


# Global model instance to avoid reloading
_model, _device = load_model()


def CNN(lidar, camera) -> tuple[float, float]:
    """
    Process lidar and camera data using a pre-trained CNN to predict distance and angle.

    Args:
        lidar (np.ndarray): numpy array of lidar data.
        camera (np.ndarray): numpy array of camera image data in (H, W, C) format.

    Returns:
        tuple[float, float]: Predicted distance and angle.
    """
    # Preprocess camera data
    cam_data = np.array(camera, dtype=np.float32) / 255.0
    # Convert (H, W, C) to (C, H, W)
    cam_data = np.transpose(cam_data, (2, 0, 1))
    cam_tensor = torch.tensor(
        cam_data, dtype=torch.float32).unsqueeze(0).to(_device)

    # Preprocess lidar data
    lidar_data = np.array(lidar, dtype=np.float32)
    # Replace infinite values and normalize
    if np.any(np.isfinite(lidar_data)):
        max_lidar_val = np.max(lidar_data[np.isfinite(lidar_data)])
        if max_lidar_val > 0:
            lidar_data[np.isinf(lidar_data)] = max_lidar_val
            lidar_data /= max_lidar_val

    lidar_tensor = torch.tensor(
        lidar_data, dtype=torch.float32).unsqueeze(0).to(_device)

    # Get prediction
    with torch.no_grad():
        output = _model(cam_tensor, lidar_tensor)
        # --- DE-NORMALIZE THE OUTPUT ---
        # The model's output is in the [-1, 1] range due to the Tanh activation.
        # We convert it back to the original scales.
        dist_normalized, angle_normalized = output[0].cpu().numpy()
        dist = dist_normalized * 6.0      # Original scale: [-6, 6]
        angle = angle_normalized * np.pi  # Original scale: [-pi, pi]

    return float(dist), float(angle)


def CNN_online(lidar_data, camera_data, dist_gt, angle_gt, learning_rate=1e-2):
    """
    Online CNN training function.
    This function performs one training step on the CNN model using the provided
    ground truth data and saves the updated model.
    """
    global _model, _device

    print("Calling CNN_online for online training...")

    # --- FIX 1: Camera Data Shape Correction ---
    # The runtime error indicates the input camera data might have its dimensions
    # swapped, e.g., (H, C, W) instead of the expected (H, W, C).
    cam_data_np = np.array(camera_data, dtype=np.float32)

    # The model's first conv layer expects 4 input channels. If the 3rd dimension
    # isn't 4 but the 2nd is, we assume they are swapped.
    if len(cam_data_np.shape) == 3 and cam_data_np.shape[2] != 4 and cam_data_np.shape[1] == 4:
        print(
            f"Warning: Camera data has shape {cam_data_np.shape}. Assuming W and C are swapped. Correcting...")
        # Transpose from (H, C, W) to (H, W, C)
        cam_data_np = np.transpose(cam_data_np, (0, 2, 1))

    # Preprocess and convert camera data to tensor
    cam_data_np = cam_data_np / 255.0
    # Convert (H, W, C) to (C, H, W)
    cam_data_tensor = np.transpose(cam_data_np, (2, 0, 1))
    cam_tensor = torch.tensor(
        cam_data_tensor, dtype=torch.float32).unsqueeze(0).to(_device)

    # --- FIX 2: Lidar Data Normalization ---
    # Preprocess lidar data to match the format used during pre-training.
    lidar_data_np = np.array(lidar_data, dtype=np.float32)
    if np.any(np.isfinite(lidar_data_np)):
        max_lidar_val = np.max(lidar_data_np[np.isfinite(lidar_data_np)])
        if max_lidar_val > 0:
            lidar_data_np[np.isinf(lidar_data_np)] = max_lidar_val
            lidar_data_np /= max_lidar_val

    lidar_tensor = torch.tensor(
        lidar_data_np, dtype=torch.float32).unsqueeze(0).to(_device)

    # --- NORMALIZE GROUND TRUTH DATA ---
    # The model is trained on targets in the [-1, 1] range, so we must
    # normalize the ground truth values before calculating the loss.
    dist_gt_normalized = dist_gt / 3.14
    angle_gt_normalized = angle_gt / np.pi

    # Prepare target tensor
    target_tensor = torch.tensor(
        [[dist_gt_normalized, angle_gt_normalized]], dtype=torch.float32).to(_device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(_model.parameters(), lr=learning_rate)

    # Perform one training step
    _model.train()  # Set model to training mode
    optimizer.zero_grad()
    outputs = _model(cam_tensor, lidar_tensor)
    loss = criterion(outputs, target_tensor)
    loss.backward()
    optimizer.step()
    _model.eval()  # Set model back to evaluation mode

    print(f"Online training step complete. Loss: {loss.item():.4f}")

    # Save the updated model
    torch.save(_model.state_dict(), MODEL_PATH)
    print(f"Online trained model saved to {MODEL_PATH}")

    # The global _model is updated, so returning it is optional
    return _model
