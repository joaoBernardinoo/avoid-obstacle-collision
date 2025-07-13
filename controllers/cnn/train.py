import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from models import LidarVisionFusionNet
from dataset import RobotDataset

# --- Constants ---
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
MODEL_PATH = SCRIPT_DIR / 'lvf.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 50
LIDAR_POINTS = 20


def train_model(model, train_loader, val_loader, num_epochs):
    """
    Trains the given model using the provided data loaders.
    """
    use_amp = DEVICE == 'cuda'
    print(f"Using device: {DEVICE}")
    print(
        f"Mixed Precision Training (AMP): {'Enabled' if use_amp else 'Disabled'}")

    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=5)
    scaler = GradScaler(enabled=use_amp)

    history = {
        'loss': [], 'val_loss': [],
        'mae': [], 'val_mae': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_train_mae = 0, 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = inputs['image'].to(DEVICE, non_blocking=True)
            lidars = inputs['lidar'].to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda",enabled=use_amp):
                outputs = model(images, lidars)
                loss = criterion(outputs, targets)
                mae = mae_criterion(outputs, targets)

            scaler.scale(loss).backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_train_mae += mae.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mae = total_train_mae / len(train_loader)
        history['loss'].append(avg_train_loss)
        history['mae'].append(avg_train_mae)

        model.eval()
        total_val_loss, total_val_mae = 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = inputs['image'].to(DEVICE, non_blocking=True)
                lidars = inputs['lidar'].to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                with autocast("cuda",enabled=use_amp):
                    outputs = model(images, lidars)
                    loss = criterion(outputs, targets)
                    mae = mae_criterion(outputs, targets)

                total_val_loss += loss.item()
                total_val_mae += mae.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mae = total_val_mae / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)

        print(
            f"Epoch {epoch+1}/{num_epochs} -> "
            f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}"
        )
        scheduler.step(avg_val_loss)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return history


def plot_history(history):
    """Plots the training and validation history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['mae'], label='Training MAE')
    ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_title('Model Mean Absolute Error (MAE)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")


if __name__ == "__main__":
    # Instantiate Model
    model = LidarVisionFusionNet(lidar_input_features=LIDAR_POINTS).to(DEVICE)

    # Create Dataset and DataLoaders
    hdf5_path = Path('./controllers/robot/cnn_dataset.h5')
    dataset = RobotDataset(hdf5_path=hdf5_path,
                           vision_transform=model.vision_transforms)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # Train the model
    history = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # Plot results
    plot_history(history)
