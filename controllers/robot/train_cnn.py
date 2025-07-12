import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Dataset


class RobotDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["img_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor([row["dist"], row["angle"]], dtype=torch.float32)
        return image, label


# Modelo CNN


class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU()
        )
        # Inferir tamanho automaticamente com um tensor fake
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 64, 64)
            dummy_out = self.features(dummy_input)
            self.flattened_size = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Config
    DATA_DIR = "controllers/robot/dados_treino"
    CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
    IMG_SIZE = (64, 64)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),  # Normaliza entre 0 e 1
    ])
    dataset = RobotDataset(CSV_PATH, DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Treinamento
    model = CNNRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Histórico de treinamento
    train_losses = []

    for epoch in range(50):
        total_loss = 0
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(loader)
        train_losses.append(train_loss)
        print(f"[Epoch {epoch}] Loss: {train_loss:.4f}")

    # Plotar o histórico de treinamento
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')

    # Salvar modelo
    torch.save(model.state_dict(), "modelo_cnn.pth")
