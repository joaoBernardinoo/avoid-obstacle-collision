import torch
from torch.utils.data import Dataset
import h5py
import os
from .constants import HDF5_SAVE_PATH

class RobotDataset(Dataset):
    def __init__(self):
        if not os.path.exists(HDF5_SAVE_PATH):
            raise FileNotFoundError(f"HDF5 dataset file not found at {HDF5_SAVE_PATH}")
        try:
            self.hf = h5py.File(HDF5_SAVE_PATH, 'r')
            self.camera_images = self.hf['camera_image']
            self.lidar_data = self.hf['lidar_data']
            self.dists = self.hf['dist']
            self.angles = self.hf['angle']
        except Exception as e:
            raise IOError(f"Error opening HDF5 file {HDF5_SAVE_PATH}: {e}")

    def __len__(self):
        return len(self.camera_images)

    def __getitem__(self, idx):
        camera_image = self.camera_images[idx]
        lidar_data = self.lidar_data[idx]
        dist = self.dists[idx]
        angle = self.angles[idx]

        # Convert to torch tensors
        camera_image = torch.from_numpy(camera_image).float()
        lidar_data = torch.from_numpy(lidar_data).float()
        dist = torch.tensor(dist).float()
        angle = torch.tensor(angle).float()

        return camera_image, lidar_data, dist, angle

    def close(self):
        if hasattr(self, 'hf') and self.hf:
            self.hf.close()

    def __del__(self):
        self.close()
