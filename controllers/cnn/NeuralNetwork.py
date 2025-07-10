import numpy as np
import torch
from pathlib import Path
import sys
# A importação do seu modelo CNN.
from .cnn_model import CNNNavigationModel

# --- MELHORIA: Centralizar constantes de normalização e desnormalização ---
# Estes valores DEVEM ser os mesmos usados durante o treino.
# Assumindo a sua premissa de que 3.14 é o correto para a distância.
DIST_SCALE = 3.14
ANGLE_SCALE = np.pi
LIDAR_MAX_VAL = 3.14

# --- Caminhos dos modelos ---
SCRIPT_DIR = Path(__file__).parent if '__file__' in locals() else Path.cwd()
MODEL_PATH = SCRIPT_DIR.parent / 'cnn' / 'final_model.pth' # Renomeado para clareza

def load_model():
    """Carrega o modelo CNN pré-treinado."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # A forma do Lidar deve corresponder ao modelo treinado (ex: 20 pontos)
    model = CNNNavigationModel(lidar_shape_in=20).to(device)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Carregado modelo pré-treinado de {MODEL_PATH}")
    else:
        # Lança um erro se o modelo não for encontrado, pois a função não pode operar sem ele.
        raise FileNotFoundError(f"Erro: Arquivo do modelo não encontrado em {MODEL_PATH}. O programa não pode continuar.")

    model.eval()
    return model, device

# Instância global do modelo para evitar recarregamentos
_model, _device = load_model()


def CNN(lidar: np.ndarray, camera: np.ndarray) -> tuple[float, float]:
    """
    Processa dados de lidar e câmera usando uma CNN para prever distância e ângulo.

    Args:
        lidar (np.ndarray): Array numpy com os dados do LiDAR.
        camera (np.ndarray): Array numpy da imagem da câmera no formato (H, W, C).

    Returns:
        tuple[float, float]: A distância e o ângulo previstos (em metros e radianos).
    """
    # 1. Pré-processamento da Câmera
    cam_data = np.array(camera, dtype=np.float32) / 255.0
    cam_data = np.transpose(cam_data, (2, 0, 1)) # Converte (H, W, C) para (C, H, W)
    cam_tensor = torch.tensor(cam_data, dtype=torch.float32).unsqueeze(0).to(_device)

    # 2. --- CORREÇÃO CRÍTICA: Pré-processamento do LiDAR ---
    # A normalização aqui deve espelhar EXATAMENTE a normalização do treino.
    lidar_data = np.array(lidar, dtype=np.float32)
    # Substitui valores infinitos pelo valor máximo conhecido
    lidar_data[np.isinf(lidar_data)] = LIDAR_MAX_VAL
    # Normaliza usando o valor máximo FIXO do dataset de treino
    lidar_data /= LIDAR_MAX_VAL
    # Garante que os valores não excedam 1.0
    lidar_data = np.clip(lidar_data, 0.0, 1.0)
    lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0).to(_device)

    # 3. Obtém a previsão do modelo
    with torch.no_grad():
        output = _model(cam_tensor, lidar_tensor)
        
        # 4. --- CORREÇÃO CRÍTICA: Desnormalização da saída ---
        # Converte a saída do modelo no intervalo [-1, 1] de volta para as escalas originais.
        dist_normalized, angle_normalized = output[0].cpu().numpy()
        
        # Multiplica pelos mesmos fatores usados para dividir durante o treino.
        dist = dist_normalized * DIST_SCALE
        angle = angle_normalized * ANGLE_SCALE

        # Descomente a linha abaixo para uma depuração fácil em tempo real
        # print(f"Saída CNN -> Norm: [D:{dist_normalized:.2f}, A:{angle_normalized:.2f}] | Des-Norm: [D:{dist:.2f}m, A:{np.degrees(angle):.2f}°]")

    return float(dist), float(angle)


