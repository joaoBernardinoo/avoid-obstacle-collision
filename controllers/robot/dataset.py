import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class RobotDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label for a given index in the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        image : PIL.Image
            The image associated with the given index.
        label : torch.Tensor
            The label associated with the given index, a tensor containing distance and angle.
        """
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([row["dist"], row["angle"]], dtype=torch.float32)
        return image, label
