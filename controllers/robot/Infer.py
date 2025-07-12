import numpy as np
import logging
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os
import cv2
from PIL import Image
import math
from typing import List
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time

logging.getLogger("pgmpy").setLevel(logging.ERROR)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bayes_model')

DATA_DIR = "dados_treino"
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, "labels.csv")
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["img_path", "dist", "angle"]).to_csv(csv_path, index=False)

DISPLAY = None
# Forçar a recriação do modelo com as novas probabilidades
if os.path.exists(MODEL_PATH):
    # model = DiscreteBayesianNetwork.load(MODEL_PATH + ".bif")
    os.remove(MODEL_PATH)
    print("[INFO] Modelo antigo removido, criando novo modelo")

if not os.path.exists(MODEL_PATH):
    # 1. Definir a estrutura da Rede Bayesiana (nós e arestas/dependências)
    # A estrutura reflete as dependências causais que definimos antes.
    model = DiscreteBayesianNetwork([
        ('ObstacleDetected', 'Action'),
        ('TargetVisible', 'Action'),
        ('Direction', 'Action'),
        # O sucesso em chegar ao destino depende da sequência de ações
        ('Action', 'Success')
    ])

    # CPT para a Ação. Depende de ObstacleDetected, TargetVisible e Direction.
    # 2. Definir as Tabelas de Probabilidade Condicional (CPTs)
    # --- CPTs para os Nós Raiz (A CORREÇÃO ESTÁ AQUI) ---
    # Usamos um valor padrão uniforme. Eles serão sobrescritos pela evidência.
    cpt_O = TabularCPD(
        variable='ObstacleDetected',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'ObstacleDetected': ['sim', 'nao']}
    )

    cpt_V = TabularCPD(
        variable='TargetVisible',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'TargetVisible': ['sim', 'nao']}
    )

    cpt_D = TabularCPD(
        variable='Direction',
        variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Direction': ['esquerda', 'frente', 'direita']}
    )
    values = [
        # seguir, v_esq, v_dir, parar
        [0.10, 0.85, 0.00, 0.05],  # ( sim, sim, esquerda )
        [0.10, 0.00, 0.00, 0.90],  # ( sim, sim, frente )
        [0.10, 0.00, 0.85, 0.05],  # ( sim, sim, direita )
        [0.10, 0.80, 0.10, 0.00],  # ( sim, nao, esquerda )
        [0.10, 0.45, 0.45, 0.00],  # ( sim, nao, frente )
        [0.10, 0.10, 0.80, 0.00],  # ( sim, nao, direita )
        [0.90, 0.00, 0.00, 0.10],  # ( nao, sim, esquerda )
        [0.90, 0.00, 0.00, 0.10],  # ( nao, sim, frente )
        [0.90, 0.00, 0.00, 0.10],  # ( nao, sim, direita )
        [0.10, 0.50, 0.30, 0.10],  # ( nao, nao, esquerda )
        [0.40, 0.25, 0.25, 0.10],  # ( nao, nao, frente )
        [0.10, 0.30, 0.50, 0.10]   # ( nao, nao, direita )
    ]

    # A ordem das evidências em 'evidence' é crucial e deve corresponder ao shape do array 'values'.
    cpt_A = TabularCPD(
        variable='Action',
        variable_card=4,  # 4 estados: seguir, v_esq, v_dir, parar
        values=np.array(values).T,
        evidence=['ObstacleDetected', 'TargetVisible', 'Direction'],
        evidence_card=[2, 2, 3],  # O(2), V(2), D(3)
        state_names={
            'Action': ['seguir', 'v_esq', 'v_dir', 'parar'],
            'ObstacleDetected': ['sim', 'nao'],
            'TargetVisible': ['sim', 'nao'],
            'Direction': ['esquerda', 'frente', 'direita']
        }
    )

    cpt_S = TabularCPD(
        variable='Success',
        variable_card=2,  # 2 estados: sim, nao
        values=[
            [0.90,   0.60,   0.60,  0.99],  # Probabilidade de sucesso 'sim'
            [0.10,   0.40,   0.40,  0.01]  # Probabilidade de sucesso 'nao'
        ],
        evidence=['Action'],
        evidence_card=[4],
        state_names={
            'Success': ['sim', 'nao'],
            'Action': ['seguir', 'v_esq', 'v_dir', 'parar']
        }
    )

    # Adicionar as CPTs ao modelo
    model.add_cpds(cpt_A, cpt_S, cpt_D, cpt_O, cpt_V)
    # Verificar se o modelo e as CPTs são consistentes

    model.save(MODEL_PATH+".bif")
    model.save(MODEL_PATH, filetype='xmlbif')
    if not model.check_model():
        raise ValueError("O modelo Bayesiano não é válido.")
    print("[INFO] Modelo Bayesiano salvo em bayes_model")


# Criar um objeto de inferência
inference = VariableElimination(model)


# make cnn return the distance to objetct and ang to target
def CNN(lidar, camera) -> tuple[float, float]:
    """
    Usa uma rede neural convolucional treinada para inferir a distância
    e ângulo entre o robô e o alvo, com base apenas na imagem da câmera.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image

    class CNNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2), nn.ReLU()
            )
            # Detecta automaticamente o tamanho da saída
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

    # Carregar modelo
    model = CNNRegressor()
    try:
        model.load_state_dict(torch.load("modelo_cnn.pth", map_location="cpu"))
        model.eval()
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo_cnn.pth: {e}")
        return 5.0, 0.0  # fallback em caso de erro

    # Preprocessamento da imagem da câmera
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    img = np.frombuffer(camera.getImage(), np.uint8).reshape((40, 200, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = Image.fromarray(img)
    x = transform(img).unsqueeze(0)  # (1, 3, 64, 64)

    with torch.no_grad():
        out = model(x).squeeze().numpy()
        dist, angle = float(out[0]), float(out[1])

    return dist, angle


def get_limits(color_bgr: List[int]):
    """
    Calcula os limites de cor HSV inferior e superior a partir de uma cor BGR de entrada.
    Esta função foi adaptada da imagem que você forneceu.
    """
    # Criamos um array 3D para a conversão, como o cvtColor espera
    c = np.uint8([[color_bgr]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Extrai o valor HSV base
    hue = hsvC[0][0][0]

    # Define os limites com base no valor de matiz (Hue)
    # Subtrai 10 do matiz para o limite inferior e adiciona 10 para o superior.
    # A saturação e o brilho são definidos para um intervalo amplo para capturar
    # a cor sob diferentes condições de iluminação.
    lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit


YELLOW = [0, 255, 255]

LOWER_LIMIT, UPPER_LIMIT = get_limits(color_bgr=YELLOW)


def probTargetVisible(image) -> float:
    # Converter para HSV

    image = np.frombuffer(
        image, np.uint8).reshape((40, 200, 4))
    image = image.copy()

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsvImage, LOWER_LIMIT, UPPER_LIMIT)

    # Apply Gaussian blur to reduce noise
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply morphological operations to further reduce noise
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = cv2.dilate(mask, kernel, iterations=2)
    # mask_ = Image.fromarray(mask)
    # bbox = mask_.getbbox()

    # if bbox is not None:
    #     x1, y1, x2, y2 = bbox
    #     # se

    #     image = cv2.rectangle(image, (x1, y1), (x2, y2),
    #                           (0, 255, 0), 1, cv2.LINE_AA)
    # resize the image with factor of 3
    mask = cv2.resize(mask, None, fx=3, fy=3)
    cv2.imshow("Webots Camera", mask)
    # Calcular proporção
    yellow_ratio = np.sum(mask > 0) / mask.size
    yellow_ratio = float(yellow_ratio)

    print(f"Proporção de pixels amarelos: {yellow_ratio:.2%}")
    # Definir probabilidade com uma transição suave de 0 a 1
    prob = 1 - np.power(2, -40 * yellow_ratio)
    return prob


def getAngle(robot_node, target) -> float:
    """Calcula o ângulo entre o robô e o obstáculo."""
    obs_pos = target.getPosition()
    rob_pos = robot_node.getPosition()
    rob_rot = robot_node.getOrientation()
    rob_yaw = math.atan2(-rob_rot[1], rob_rot[0])
    dx = obs_pos[0] - rob_pos[0]
    dy = obs_pos[1] - rob_pos[1]
    angle = math.atan2(dy, dx) - rob_yaw
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return -angle


REPULSE = "cos"
VISION = True
DIST_NEAR = 0.6
ANGLE_FRONT = 0.0218  # 15 degrees in radians


def GPS(robot_node, lidar, target):
    """
    Calcula a distância e o ângulo entre o robô e o obstáculo mais próximo.

    Parameters
    ----------
    robot_node : Supervisor
        Nó do robô.
    lidar : Lidar
        Sensor de distância do robô.
    target : Node
        Nó do obstáculo.

    Returns
    -------
    dist : float
        Distância do obstáculo mais próximo.
    angle : float
        Ângulo entre o robô e o obstáculo mais próximo.
    """
    lidar_data = lidar.getRangeImage()  # type: List[int]
    # Posição do obstáculo mais próximo
    min_dist = min(lidar_data)
    min_index = lidar_data.index(min_dist)

    angle = 0.0
    angle = getAngle(robot_node, target)
    if min_dist > DIST_NEAR:
        return min_dist, angle
    # min_dist é a distancia do ojbeto mais proximo, eu quero que
    # objetos a esquerda estejam com distancia negativa e direita com distancia positiva
    # se ele estiver no centro deve ser
    if min_index < len(lidar_data) // 2:
        min_dist = -min_dist

    return min_dist, angle


def mapSoftEvidence(robot_node, lidar, camera, target):
    """
    Versão final para NAVEGAÇÃO AUTÔNOMA.
    Esta função usa a CNN treinada para inferir a situação e
    gerar a evidência para a Rede Bayesiana.
    """
    # 1. Usar a CNN para inferir distância e ângulo a partir da imagem
    # A função CNN carrega o modelo 'modelo_cnn.pth' e faz a previsão.
    dist_estimada, angle_estimado = CNN(lidar, camera) #

    # Imprime os valores que a CNN está "vendo" para debug
    print(f"CNN -> Dist: {dist_estimada:.2f}, Ângulo: {angle_estimado*180/np.pi:.1f}°")

    # 2. Calcular a Evidência Virtual com base na saída da CNN
    # Probabilidade de obstáculo (baseada na distância estimada pela CNN)
    p_obs_sim = 1 / (1 + np.exp((abs(dist_estimada) - DIST_NEAR) * 5)) #
    p_obs = [p_obs_sim, 1 - p_obs_sim]

    # Probabilidade de alvo visível (baseada no ângulo estimado pela CNN)
    if VISION:
        p_vis_sim = probTargetVisible(camera.getImage()) #
    else:
        p_vis_sim = 1.0 if abs(angle_estimado) < 0.7854 else 0.1 #
    p_vis = [p_vis_sim, 1 - p_vis_sim]

    # Cálculo da direção (baseado na distância e ângulo estimados pela CNN)
    if angle_estimado < -ANGLE_FRONT: #
        p_dir_target = np.array([0.8, 0.1, 0.1]) #
    elif angle_estimado > ANGLE_FRONT: #
        p_dir_target = np.array([0.1, 0.1, 0.8]) #
    else:
        p_dir_target = np.array([0.1, 0.8, 0.1]) #

    if dist_estimada < 0: #
        p_dir_avoidance = np.array([0.1, 0.2, 0.7]) #
    else:
        p_dir_avoidance = np.array([0.7, 0.2, 0.1]) #

    p_dir = p_dir_target * (1 - p_obs_sim) + p_dir_avoidance * p_obs_sim #

    # 3. Montar e retornar a evidência para a Rede Bayesiana
    virtual_evidence = [
        TabularCPD('ObstacleDetected', 2, [[p_obs[0]], [p_obs[1]]],
                   state_names={'ObstacleDetected': ['sim', 'nao']}), #
        TabularCPD('TargetVisible', 2, [[p_vis[0]], [p_vis[1]]],
                   state_names={'TargetVisible': ['sim', 'nao']}), #
        TabularCPD('Direction', 3, [[p_dir[0]], [p_dir[1]], [p_dir[2]]],
                   state_names={'Direction': ['esquerda', 'frente', 'direita']}) #
    ]

    # A função não imprime mais a evidência, apenas a retorna
    return virtual_evidence



def bayesian(soft_evidence) -> tuple[str, float]:
    """Usa o modelo pgmpy para inferir a ação e a probabilidade de sucesso com soft evidence."""
    action_map = {0: 'seguir', 1: 'v_esq', 2: 'v_dir', 3: 'parar'}

    # Inferir a distribuição de Action com soft evidence
    action_dist = inference.query(
        # type: ignore
        variables=['Action'], virtual_evidence=soft_evidence)

    # debug
    print(action_dist)

    action_idx = int(np.argmax(action_dist.values))  # type: ignore
    action_str = action_map[action_idx]

    # # Probabilidade de sucesso para a ação escolhida
    # prob_success_dist = inference.query(
    #     variables=['Success'],
    #     evidence={'Action': action_str}  # Use state name instead of index
    # ) # type: ignore
    # p_success = prob_success_dist.values[0] # type: ignore
    p_success = 0.0
    return action_str, p_success  # type: ignore

