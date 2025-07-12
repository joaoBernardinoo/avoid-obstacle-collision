import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import CNNRegressor
from dataset import RobotDataset

# --- Evaluation Script ---

def evaluate_model():
    # Config
    DATA_DIR = "controllers/robot/dados_treino"
    CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
    IMG_SIZE = (64, 64)
    MODEL_PATH = "modelo_cnn.pth"
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    # Load model
    model = CNNRegressor()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load dataset
    dataset = RobotDataset(CSV_PATH, DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    reals = []
    preds = []

    # Get predictions
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            reals.append(y.numpy())
            preds.append(pred.numpy())

    reals = np.concatenate(reals)
    preds = np.concatenate(preds)

    real_dist, real_angle = reals[:, 0], reals[:, 1]
    pred_dist, pred_angle = preds[:, 0], preds[:, 1]

    # Calculate errors
    error_dist = real_dist - pred_dist
    error_angle = real_angle - pred_angle

    print(f"Distance - MSE: {mean_squared_error(real_dist, pred_dist):.4f}")
    print(f"Angle - MSE: {mean_squared_error(real_angle, pred_angle):.4f}")

    # --- Accuracy Calculation ---
    dist_tolerance = 0.1
    angle_tolerance_deg = 1.0
    angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)

    correct_dist_preds = np.sum(np.abs(error_dist) < dist_tolerance)
    correct_angle_preds = np.sum(np.abs(error_angle) < angle_tolerance_rad)

    dist_accuracy = (correct_dist_preds / len(real_dist)) * 100
    angle_accuracy = (correct_angle_preds / len(real_angle)) * 100

    print(f"\nDistance Accuracy (tolerance +/- {dist_tolerance}): {dist_accuracy:.2f}%")
    print(f"Angle Accuracy (tolerance +/- {angle_tolerance_deg:.1f} degree): {angle_accuracy:.2f}%")


    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Predição vs Real - Distância
    axs[0, 0].scatter(real_dist, pred_dist, alpha=0.6)
    axs[0, 0].plot([real_dist.min(), real_dist.max()], [real_dist.min(), real_dist.max()], 'r--')
    axs[0, 0].set_xlabel("Distância Real")
    axs[0, 0].set_ylabel("Distância Predita")
    axs[0, 0].set_title("Predição vs Real - Distância")
    axs[0, 0].grid(True)

    # Predição vs Real - Ângulo
    axs[0, 1].scatter(real_angle, pred_angle, alpha=0.6)
    axs[0, 1].plot([real_angle.min(), real_angle.max()], [real_angle.min(), real_angle.max()], 'r--')
    axs[0, 1].set_xlabel("Ângulo Real")
    axs[0, 1].set_ylabel("Ângulo Predito")
    axs[0, 1].set_title("Predição vs Real - Ângulo")
    axs[0, 1].grid(True)

    # Distribuição do Erro - Distância
    axs[1, 0].hist(error_dist, bins=50, alpha=0.7)
    axs[1, 0].axvline(error_dist.mean(), color='r', linestyle='--')
    axs[1, 0].set_xlabel("Erro (Real - Predito)")
    axs[1, 0].set_ylabel("Frequência")
    axs[1, 0].set_title("Distribuição do Erro - Distância")
    axs[1, 0].grid(True)

    # Distribuição do Erro - Ângulo
    axs[1, 1].hist(error_angle, bins=50, alpha=0.7)
    axs[1, 1].axvline(error_angle.mean(), color='r', linestyle='--')
    axs[1, 1].set_xlabel("Erro (Real - Predito)")
    axs[1, 1].set_ylabel("Frequência")
    axs[1, 1].set_title("Distribuição do Erro - Ângulo")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("\nPlot saved to evaluation_results.png")


if __name__ == "__main__":
    evaluate_model()
