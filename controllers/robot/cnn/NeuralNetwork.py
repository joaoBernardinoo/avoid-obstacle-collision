# import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import torch
import cv2
import math
from constants import LOWER_LIMIT, UPPER_LIMIT


# Supondo que a classe LidarVisionFusionNet esteja em um arquivo models.py
# na mesma pasta ou em um caminho acessível pelo Python.
# Esta estrutura é consistente com seus scripts train.py e eval.py.
if True:
    sys.path.append(str(Path(__file__).parent))
    from models import BallAngleCNN

# --- Configuração de Caminhos e Constantes ---
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# O caminho do modelo deve ser o mesmo usado no script de treinamento (train.py)
MODEL_PATH = SCRIPT_DIR / 'angle.pth'
LIDAR_POINTS = 20  # Consistente com o script de treinamento
# Defina o alcance máximo do seu sensor LiDAR para normalização.
# Este valor deve ser o mesmo usado durante o treinamento para melhores resultados.
LIDAR_MAX_RANGE = 3.14  # Exemplo: 5 metros


def load_model():
    """Carrega o modelo LidarVisionFusionNet pré-treinado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Instancia o modelo correto
    model = BallAngleCNN(1).to(device)

    if MODEL_PATH.exists():
        # Carrega os pesos do modelo treinado
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Modelo pré-treinado carregado de {MODEL_PATH}")
    else:
        print(f"Aviso: Arquivo do modelo não encontrado em {MODEL_PATH}.")
        print("Usando um modelo não treinado. As previsões não serão úteis.")

    # Coloca o modelo em modo de avaliação (importante para desativar dropout, etc.)
    model.eval()
    return model, device


# --- Instância Global do Modelo ---
# Carrega o modelo uma vez para evitar recarregá-lo a cada chamada da função
_model, _device = load_model()
# Obtém as transformações de imagem do modelo carregado


def CNN(camera: np.ndarray) -> float:
    """
    Processa dados de Lidar e Câmera usando a rede neural para prever distância e ângulo.

    Args:
        camera (np.ndarray): Array numpy com a imagem da câmera no formato (H, W, C).

    Returns:
        tuple[float, float]: Uma tupla contendo a distância e o ângulo previstos.
    """
    with torch.no_grad():
        # 1. Pré-processamento da Imagem
        # # Converte BGR para grayscale (se a imagem for colorida)

        bgr_weights = [0.1140, 0.5870, 0.2989]
        camera = camera[:, :, :3]
        image_gray = np.dot(camera, bgr_weights)
        print(image_gray.shape)

        # Normaliza para [0, 1] e converte para float32
        image = image_gray.astype(np.float32) / 255.0
        # image = camera.astype(np.float32) / 255.0
        # Converte para tensor, adiciona dimensões de canal e batch
        camera_tensor = torch.from_numpy(image) \
            .unsqueeze(0) \
            .unsqueeze(0) \
            .to(_device)

        # 2. Predição da Rede Neural
        prediction = _model(camera_tensor)

        # 3. Pós-processamento da Saída
        angle = float(prediction.squeeze().cpu().numpy())

        return angle
