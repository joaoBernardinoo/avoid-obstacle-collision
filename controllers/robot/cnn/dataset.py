import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from PIL import Image
import numpy as np


class RobotDataset(Dataset):
    """
    Dataset otimizado para carregar dados de robótica de um arquivo HDF5.

    - Carrega dados "on-the-fly" para economizar RAM.
    - Retorna dados em um formato limpo (inputs, targets) para o DataLoader.
    """

    def __init__(self, hdf5_path: Path, vision_transform=None):
        self.hdf5_path = hdf5_path
        self.vision_transform = vision_transform

        self._file = None
        # Abre o arquivo uma vez para obter o tamanho e evitar reaberturas
        with h5py.File(self.hdf5_path, 'r') as f:
            self._len = f['camera_image'].shape[0]  # type: ignore

    def __len__(self) -> int:
        return self._len

   # Dentro da sua classe RobotDatasetV2

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')

        # Carrega os dados do disco
        image_data = self._file['camera_image'][idx][:, :, :3]  # <-- Isso é um array NumPy  # type: ignore
        lidar_data = self._file['lidar_data'][idx]  # type: ignore
        dist_data = self._file['dist'][idx]  # type: ignore
        angle_data = self._file['angle'][idx]  # type: ignore

        # --- CORREÇÃO AQUI ---
        # Converta o array NumPy para uma Imagem PIL antes de transformar
        lidar_tensor = torch.from_numpy(lidar_data).float()

        # Handle NaNs and infinity, then normalize
        lidar_data = np.nan_to_num(lidar_data, nan=0.0, posinf=3.14, neginf=-3.14)  # type: ignore
        lidar_data /= 3.14  # Normalize to [-1, 1]

        # 2. Converte para Tensores e aplica transformações
        if self.vision_transform:
            image_tensor = self.vision_transform(image_data)
        else:
            # Fallback caso nenhuma transform seja passada
            image_tensor = torch.from_numpy(image_data).permute(2, 0, 1).float() / 255.0

        lidar_tensor = torch.from_numpy(lidar_data).float()
        targets = torch.tensor([dist_data, angle_data], dtype=torch.float32)
        inputs = {'image': image_tensor, 'lidar': lidar_tensor}

        return inputs, targets
