import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import CNNRegressor
from dataset import RobotDataset


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
