import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# It's assumed that your model and dataset classes are defined in these files
from model import CNNRegressor
from dataset import RobotDataset


if __name__ == "__main__":
    # --- Device Configuration --- ⚙️
    # Selects the GPU if CUDA is available, otherwise defaults to the CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Config ---
    DATA_DIR = "controllers/robot/dados_treino"
    MODEL_PATH = "modelo_cnn.pth"
    CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
    IMG_SIZE = (64, 64)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),  # Normalizes to the range [0, 1]
    ])

    dataset = RobotDataset(CSV_PATH, DATA_DIR, transform)

    # --- Data Splitting ---
    validation_split = 0.2
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Using num_workers can speed up data loading, especially with a GPU
    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            pin_memory=True)

    # --- Model Initialization and Loading ---
    model = CNNRegressor()
    exit = False
    # Check if a pre-trained model exists and load its state
    if os.path.exists(MODEL_PATH):
        # pergunta se o usuario quer deletar o modelo atual
        print("Pre-trained model found. Do you want to delete it? (y/n)")
        if input() == 'n':
            pass
        else:
            exit = True
            print("Deleting pre-trained model...")
            os.remove(MODEL_PATH)
        if not exit:
            print(f"Pre-trained model found at '{MODEL_PATH}'. Loading weights...")
            # Load weights onto the correct device
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("No pre-trained model found. Starting training from scratch.")

    # Move the model to the selected device (GPU or CPU)
    model.to(device)

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    num_epochs = 50

    print(f"Starting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            # Move the batch of data (images and labels) to the selected device
            x = x.to(device)
            y = y.to(device)


            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                total_val_loss += loss_fn(pred, y).item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(
            f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # --- Plotting and Saving ---
    print("Training finished. Saving model and plotting history...")

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    # Save the final model state
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model successfully saved to '{MODEL_PATH}'")
