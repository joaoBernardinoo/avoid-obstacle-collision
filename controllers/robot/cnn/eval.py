import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
from pathlib import Path

from models import LidarVisionFusionNet
from dataset import RobotDataset
from train import LIDAR_POINTS, MODEL_PATH, DEVICE, BATCH_SIZE


def evaluate_model(model, val_loader):
    model.eval()
    reals, preds = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            images = inputs['image'].to(DEVICE)
            lidars = inputs['lidar'].to(DEVICE)

            outputs = model(images, lidars)

            preds.append(outputs.cpu().numpy())
            reals.append(targets.cpu().numpy())

    # Concatenate all batches
    reals = np.concatenate(reals, axis=0)
    preds = np.concatenate(preds, axis=0)

    real_dist, real_angle = reals[:, 0], reals[:, 1]
    pred_dist, pred_angle = preds[:, 0], preds[:, 1]

    # Calculate metrics
    mse_dist = mean_squared_error(real_dist, pred_dist)
    mae_dist = mean_absolute_error(real_dist, pred_dist)
    mse_angle = mean_squared_error(real_angle, pred_angle)
    mae_angle = mean_absolute_error(real_angle, pred_angle)

    print(f"Distance - MSE: {mse_dist:.4f}, MAE: {mae_dist:.4f}")
    print(f"Angle - MSE: {mse_angle:.4f}, MAE: {mae_angle:.4f}")

    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    # Distance plot
    axs[0, 0].scatter(real_dist, pred_dist, alpha=0.6, label='Predictions')
    axs[0, 0].plot([real_dist.min(), real_dist.max()], [
                   real_dist.min(), real_dist.max()], 'r--', label='Ideal')
    axs[0, 0].set_xlabel("Real Distance")
    axs[0, 0].set_ylabel("Predicted Distance")
    axs[0, 0].set_title("Distance: Prediction vs. Real")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Angle plot
    axs[0, 1].scatter(real_angle, pred_angle, alpha=0.6, label='Predictions')
    axs[0, 1].plot([real_angle.min(), real_angle.max()], [
                   real_angle.min(), real_angle.max()], 'r--', label='Ideal')
    axs[0, 1].set_xlabel("Real Angle")
    axs[0, 1].set_ylabel("Predicted Angle")
    axs[0, 1].set_title("Angle: Prediction vs. Real")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Distance error distribution
    axs[1, 0].hist(real_dist - pred_dist, bins=30, alpha=0.7)
    axs[1, 0].set_xlabel("Prediction Error")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Distance Error Distribution")
    axs[1, 0].grid(True)

    # Angle error distribution
    axs[1, 1].hist(real_angle - pred_angle, bins=30, alpha=0.7)
    axs[1, 1].set_xlabel("Prediction Error")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_title("Angle Error Distribution")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("\nPlot saved to evaluation_results.png")


if __name__ == "__main__":
    # Load model
    model = LidarVisionFusionNet(lidar_input_features=LIDAR_POINTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Create Dataset and DataLoaders
    hdf5_path = Path('./controllers/robot/cnn_dataset.h5')
    dataset = RobotDataset(hdf5_path=hdf5_path,
                           vision_transform=model.vision_transforms)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    evaluate_model(model, val_loader)
