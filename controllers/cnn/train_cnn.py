#!/usr/bin/env python3
import sys
import os
import h5py
from cnn_model import CNNNavigationModel
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')  # Define o backend não-interativo explicitamente

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
if sys.platform == 'linux':
    DATASET_PATH = Path('/home/dino/SSD/cnn_dataset.h5')
else:
    DATASET_PATH = Path(os.path.dirname(__file__), 'cnn_dataset')
# Diretório para salvar os checkpoints
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
PLOT_SAVE_PATH = SCRIPT_DIR / 'training_history.png'

# Cria o diretório de checkpoints se ele não existir
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Image dimensions
IMG_HEIGHT = 40
IMG_WIDTH = 200
IMG_CHANNELS = 4  # BGRA from Webots

# Training parameters
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
N_SPLITS = 5  # Número de folds para K-Fold

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(path: Path, dataset_type: str, max_samples=None):
    """
    Carrega o dataset de arquivos .npz ou de um único arquivo HDF5.
    """
    if dataset_type == 'npz':
        cam_images, lidar_data, targets = [], [], []

        if not path.exists() or not path.is_dir():
            print(f"Error: Dataset directory not found at '{path}'")
            raise FileNotFoundError

        file_list = sorted(list(path.glob('*.npz')))
        if not file_list:
            print(f"Error: No .npz files found in '{path}'")
            raise ValueError

        if max_samples:
            file_list = file_list[:max_samples]

        print(f"Loading {len(file_list)} samples from '{path}'...")
        for f_path in tqdm(file_list):
            with np.load(f_path) as data:
                cam_images.append(data['camera_image'])
                lidar_data.append(data['lidar_data'])
                targets.append([data['dist'], data['angle']])

        # Converte para arrays numpy
        X_cam = np.array(cam_images, dtype=np.float32)
        X_lidar = np.array(lidar_data, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

    elif dataset_type == 'hdf5':
        if not path.exists() or not path.is_file():
            print(f"Error: HDF5 file not found at '{path}'")
            raise FileNotFoundError

        with h5py.File(path, 'r') as hf:
            if 'camera_image' not in hf:
                print(f"Error: 'camera_image' dataset not found in '{path}'")
                raise ValueError

            print(f"Loading samples from '{path}'...")
            X_cam = np.array(hf['camera_image'], dtype=np.float32)
            X_lidar = np.array(hf['lidar_data'], dtype=np.float32)
            dist = np.array(hf['dist'], dtype=np.float32)
            angle = np.array(hf['angle'], dtype=np.float32)
            y = np.stack((dist, angle), axis=1)

            if max_samples:
                X_cam = X_cam[:max_samples]
                X_lidar = X_lidar[:max_samples]
                y = y[:max_samples]

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # Normaliza imagens
    X_cam /= 255.0

    # --- NORMALIZE TARGETS (y) ---
    y[:, 0] /= 3.14  # Normalize distance
    y[:, 1] /= np.pi   # Normalize angle
    y = np.clip(y, -1.0, 1.0)

    # Normaliza dados do LiDAR
    if X_lidar.size > 0:
        max_lidar_val = np.max(X_lidar[np.isfinite(X_lidar)])
        if max_lidar_val > 0:
            X_lidar[np.isinf(X_lidar)] = max_lidar_val
            X_lidar /= max_lidar_val

    # Permuta os eixos da imagem para o formato do PyTorch (N, C, H, W)
    X_cam = np.transpose(X_cam, (0, 3, 1, 2))

    return (X_cam, X_lidar), y


# Importa o modelo de um arquivo separado


def plot_history(history):
    """Plota o histórico de treinamento e validação e salva em um arquivo."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history['mae'], label='Training MAE')
    ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)  # Salva a figura em um arquivo
    plt.close(fig)  # Fecha a figura para liberar memória
    print(f"Training plot saved to '{PLOT_SAVE_PATH}'")
    plt.close(fig)  # Fecha a figura para liberar memória


def fine_tune_model(best_checkpoint_path: Path, full_dataset_path: Path, dataset_type: str, fine_tune_epochs: int = 10, fine_tune_lr: float = 1e-5):
    print(f"\n--- Starting Fine-tuning on Entire Dataset ---")
    print(f"Loading best model from: {best_checkpoint_path}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the full dataset
    (X_cam, X_lidar), y = load_model(full_dataset_path, dataset_type)
    if X_cam is None or X_cam.shape[0] == 0:
        print("Exiting fine-tuning: No data loaded for full dataset.")
        return

    print(
        f"Full dataset loaded. Shapes: Cam={X_cam.shape}, Lidar={X_lidar.shape}, Target={y.shape}")

    # Create DataLoader for the full dataset
    full_dataset = TensorDataset(torch.tensor(
        X_cam), torch.tensor(X_lidar), torch.tensor(y))
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model and load state dict
    model = CNNNavigationModel(lidar_shape_in=X_lidar.shape[1]).to(device)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer for fine-tuning (can be a new one with lower LR)
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    # Optionally load optimizer state if resuming fine-tuning
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    print(f"Fine-tuning for {fine_tune_epochs} epochs with LR: {fine_tune_lr}")

    model.train()  # Set model to training mode
    for epoch in range(fine_tune_epochs):
        running_loss, running_mae = 0.0, 0.0
        progress_bar = tqdm(
            full_loader, desc=f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}", leave=False)
        for cam_batch, lidar_batch, target_batch in progress_bar:
            cam_batch, lidar_batch, target_batch = cam_batch.to(
                device), lidar_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            outputs = model(cam_batch, lidar_batch)
            loss = criterion(outputs, target_batch)
            mae = mae_criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_mae += mae.item()
            progress_bar.set_postfix(loss=loss.item(), mae=mae.item())

        epoch_loss = running_loss / len(full_loader)
        epoch_mae = running_mae / len(full_loader)
        print(
            f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs} - Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}")

    # Save the final fine-tuned model
    final_model_path = SCRIPT_DIR / 'final_cnn_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Fine-tuned model saved to: {final_model_path}")


if __name__ == "__main__":
    # Altere o 'dataset_type' para 'npz' se quiser carregar o dataset antigo
    (X_cam, X_lidar), y = load_model(DATASET_PATH, dataset_type='hdf5')

    if X_cam is None or X_cam.shape[0] == 0:
        print("Exiting: No data loaded.")
        exit()

    print(
        f"Data loaded. Shapes: Cam={X_cam.shape}, Lidar={X_lidar.shape}, Target={y.shape}")

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_histories = []

    best_overall_val_loss = float('inf')
    best_overall_checkpoint_path = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_cam)):
        print(f"\n--- Starting Fold {fold+1}/{N_SPLITS} ---")

        # Define o caminho do checkpoint para este fold específico
        checkpoint_path = CHECKPOINT_DIR / f'checkpoint_fold_{fold+1}.pth'

        # Split data for this fold
        X_cam_train, y_train = X_cam[train_idx], y[train_idx]
        X_lidar_train = X_lidar[train_idx]

        X_cam_val, y_val = X_cam[val_idx], y[val_idx]
        X_lidar_val = X_lidar[val_idx]

        # Converte dados para tensores PyTorch
        train_dataset = TensorDataset(torch.tensor(
            X_cam_train), torch.tensor(X_lidar_train), torch.tensor(y_train))
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(
            X_cam_val), torch.tensor(X_lidar_val), torch.tensor(y_val))
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Instancia o modelo e o otimizador
        model = CNNNavigationModel(lidar_shape_in=X_lidar.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Inicializa variáveis para o loop de treino e checkpoint
        start_epoch = 0
        best_val_loss = float('inf')
        history = {'loss': [], 'val_loss': [], 'mae': [],
                   'val_mae': [], "best_val_mae": 0.0}

        # --- LÓGICA PARA CARREGAR CHECKPOINT ---
        if checkpoint_path.exists():
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            history = checkpoint['history']
            print(
                f"Resumed from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
        else:
            print(
                f"No checkpoint found for fold {fold+1}. Starting from scratch.")

        # Define as funções de perda
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        print(
            f"\n--- Starting Model Training for Fold {fold+1} from Epoch {start_epoch+1} ---")
        for epoch in range(start_epoch, EPOCHS):
            # --- Fase de Treinamento ---
            model.train()
            running_loss, running_mae = 0.0, 0.0
            progress_bar = tqdm(
                train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS} [T]", leave=False)
            for cam_batch, lidar_batch, target_batch in progress_bar:
                cam_batch, lidar_batch, target_batch = cam_batch.to(
                    device), lidar_batch.to(device), target_batch.to(device)
                optimizer.zero_grad()
                outputs = model(cam_batch, lidar_batch)
                loss = criterion(outputs, target_batch)
                mae = mae_criterion(outputs, target_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_mae += mae.item()
                progress_bar.set_postfix(loss=loss.item(), mae=mae.item())

            epoch_loss = running_loss / len(train_loader)
            epoch_mae = running_mae / len(train_loader)
            history['loss'].append(epoch_loss)
            history['mae'].append(epoch_mae)

            # --- Fase de Validação ---
            model.eval()
            val_loss, val_mae = 0.0, 0.0
            with torch.no_grad():
                for cam_batch, lidar_batch, target_batch in val_loader:
                    cam_batch, lidar_batch, target_batch = cam_batch.to(
                        device), lidar_batch.to(device), target_batch.to(device)
                    outputs = model(cam_batch, lidar_batch)
                    val_loss += criterion(outputs, target_batch).item()
                    val_mae += mae_criterion(outputs, target_batch).item()

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_mae = val_mae / len(val_loader)
            history['val_loss'].append(epoch_val_loss)
            history['val_mae'].append(epoch_val_mae)

            print(
                f"Fold {fold+1}/{N_SPLITS} Epoch {epoch+1}/{EPOCHS} - "
                f"Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f} - "
                f"Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}"
            )

            # --- LÓGICA PARA SALVAR CHECKPOINT ---
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                print(
                    f"  -> New best validation loss: {best_val_loss:.4f}. Saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_mae': epoch_val_mae,
                    'history': history
                }, checkpoint_path)

        history['best_val_mae'] = epoch_val_mae
        fold_histories.append(history)

        # Update best overall checkpoint if this fold performed better
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            current_fold_best_val_loss = checkpoint['best_val_loss']
            if current_fold_best_val_loss < best_overall_val_loss:
                best_overall_val_loss = current_fold_best_val_loss
                best_overall_checkpoint_path = checkpoint_path

    # Agrega e plota os resultados de todos os folds
    if fold_histories:
        avg_history = {
            'loss': np.mean([h['loss'] for h in fold_histories], axis=0).tolist(),
            'val_loss': np.mean([h['val_loss'] for h in fold_histories], axis=0).tolist(),
            'mae': np.mean([h['mae'] for h in fold_histories], axis=0).tolist(),
            'val_mae': np.mean([h['val_mae'] for h in fold_histories], axis=0).tolist(),
            'best_val_mae': np.mean([h['best_val_mae'] for h in fold_histories]).tolist()
        }
        plot_history(avg_history)

    print("\nTraining complete.")

    # Fine-tune the best model on the entire dataset
    if best_overall_checkpoint_path:
        # Assuming hdf5 for full dataset
        fine_tune_model(best_overall_checkpoint_path, DATASET_PATH, 'hdf5')
    else:
        print("\nNo best model found for fine-tuning.")
