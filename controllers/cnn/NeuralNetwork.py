# import torch
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Supondo que a classe LidarVisionFusionNet esteja em um arquivo models.py
# na mesma pasta ou em um caminho acessível pelo Python.
# Esta estrutura é consistente com seus scripts train.py e eval.py.
if True:
    sys.path.append(str(Path(__file__).parent.parent))
    from cnn.models import LidarVisionFusionNet

# --- Configuração de Caminhos e Constantes ---
SCRIPT_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# O caminho do modelo deve ser o mesmo usado no script de treinamento (train.py)
MODEL_PATH = SCRIPT_DIR / 'lvf.pth'
LIDAR_POINTS = 20  # Consistente com o script de treinamento
# Defina o alcance máximo do seu sensor LiDAR para normalização.
# Este valor deve ser o mesmo usado durante o treinamento para melhores resultados.
LIDAR_MAX_RANGE = 3.14  # Exemplo: 5 metros


# def load_model():
#     """Carrega o modelo LidarVisionFusionNet pré-treinado."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Usando dispositivo: {device}")

#     # Instancia o modelo correto
#     model = LidarVisionFusionNet(lidar_input_features=LIDAR_POINTS).to(device)

#     if MODEL_PATH.exists():
#         # Carrega os pesos do modelo treinado
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         print(f"Modelo pré-treinado carregado de {MODEL_PATH}")
#     else:
#         print(f"Aviso: Arquivo do modelo não encontrado em {MODEL_PATH}.")
#         print("Usando um modelo não treinado. As previsões não serão úteis.")

#     # Coloca o modelo em modo de avaliação (importante para desativar dropout, etc.)
#     model.eval()
#     return model, device


# # --- Instância Global do Modelo ---
# # Carrega o modelo uma vez para evitar recarregá-lo a cada chamada da função
# _model, _device = load_model()
# # Obtém as transformações de imagem do modelo carregado
# _vision_transforms = _model.vision_transforms


def CNN(lidar: np.ndarray, camera: np.ndarray) -> tuple[float, float]:
    """
    Processa dados de Lidar e Câmera usando a rede neural para prever distância e ângulo.

    Args:
        lidar (np.ndarray): Array numpy com os dados do LiDAR (20 pontos).
        camera (np.ndarray): Array numpy com a imagem da câmera no formato (H, W, C).

    Returns:
        tuple[float, float]: Uma tupla contendo a distância e o ângulo previstos.
    """
    # Garante que o modelo não tentará calcular gradientes, economizando memória e tempo.
    # with torch.no_grad():
    #     # 1. Pré-processamento da Imagem da Câmera
    #     # Converte o array numpy (H, W, C) para uma Imagem PIL
    #     camera_pil = Image.fromarray(camera).convert('RGB')
    #     # Aplica as transformações (redimensionamento, normalização, etc.)
    #     # e adiciona uma dimensão de batch (1, C, H, W)
    #     image_tensor = _vision_transforms(camera_pil).unsqueeze(0).to(_device)

    #     # 2. Pré-processamento dos Dados do LdividindoiDAR
    #     lidar_np = np.array(lidar)
    #     # Normaliza os dados do LiDAR dividindo pelo alcance máximo
    #     # Isso escala os valores para o intervalo [0, 1]
    #     lidar_np = np.clip(lidar_np, 0, LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE
    #     # Converte o array numpy para um tensor float, adiciona uma dimensão de batch (1, 20)
    #     lidar_tensor = torch.from_numpy(
    #         lidar_np).float().unsqueeze(0).to(_device)

    #     # 3. Executa a Inferência
    #     prediction = _model(image_tensor, lidar_tensor)

    #     # 4. Pós-processamento da Saída
    #     # Remove a dimensão de batch, move para a CPU e converte para numpy
    #     output = prediction.squeeze(0).cpu().numpy()

    #     dist = float(output[0])
    #     angle = float(output[1])

        # return dist, angle
    return 0,0
