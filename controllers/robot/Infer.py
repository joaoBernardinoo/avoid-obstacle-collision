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

logging.getLogger("pgmpy").setLevel(logging.ERROR)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bayes_model')

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
        [0.60, 0.15, 0.10, 0.15],  # ( sim, sim, esquerda )
        [0.50, 0.10, 0.20, 0.20],  # ( sim, sim, frente )
        [0.60, 0.10, 0.15, 0.15],  # ( sim, sim, direita )
        [0.10, 0.80, 0.10, 0.00],  # ( sim, nao, direita )
        [0.10, 0.45, 0.45, 0.00],  # ( sim, nao, frente )
        [0.10, 0.10, 0.80, 0.00],  # ( sim, nao, esquerda )
        [0.70, 0.15, 0.15, 0.00],  # ( nao, sim, esquerda )
        [0.60, 0.20, 0.20, 0.00],  # ( nao, sim, frente )
        [0.10, 0.10, 0.80, 0.00],  # ( nao, sim, direita )
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

    return 5.0, 0.0  # Simula uma distância de 5 metros e ângulo de 0 radianos

# gets a image and return the prob of target visible


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


def probTargetVisible(image) -> float:
    # Converter para HSV
    image = np.array(image, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir faixa de amarelo (ajuste conforme o ambiente!)
    lower_yellow, upper_yellow = get_limits(color_bgr=[0, 255, 255])

    # Máscara para regiões amarelas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Calcular proporção
    yellow_ratio = np.sum(mask > 0) / mask.size
    yellow_ratio = float(yellow_ratio)

    print(f"Proporção de pixels amarelos: {yellow_ratio:.2%}")

    # Definir probabilidade com uma transição suave
    # Usar uma função linear para mapear a proporção de amarelo para uma probabilidade entre 0.1 e 1.0
    if yellow_ratio <= 0.05:
        return 0.0
    elif yellow_ratio >= 0.40:
        return 1.0
    else:
        # Escala linear entre 0.05 e 0.20
        return 0.1 + (yellow_ratio - 0.05) * (0.9 / 0.15)


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
    if min_dist > DIST_NEAR:
        angle = getAngle(robot_node, target)
        return min_dist, angle
    # min_dist é a distancia do ojbeto mais proximo, eu quero que
    # objetos a esquerda estejam com distancia negativa e direita com distancia positiva
    # se ele estiver no centro deve ser
    if min_index < len(lidar_data) // 2:
        min_dist = -min_dist

    return min_dist, angle


def mapSoftEvidence(robot_node, lidar, camera, target):

    # TAREFA- Victor Sales
    # Inferir a distancia e o angulo entre o robô e o alvo
    # use a funcao GPS como ground truth

    # lidar_data = lidar.getRangeImage() #[0.1,0.2, 3.0 ....]
    # camera_data = camera.getImageArray() # (shape camera_w, camera_h, 3)

    # adicione essa linha quando a CNN estiver pronta
    # dist,angle = CNN(lidar_data,camera_data)

    # remover essa linha abaixo quando tiver o CNN
    dist, angle = GPS(robot_node, lidar, target)
    print("Angulo Bolinha Amarela", angle * 180 / np.pi)
    print("Distancia Objeto Mais Próximo", dist)

    # Soft evidence (probabilidades)
    p_obs_sim = 1 / (1 + np.exp((abs(dist) - DIST_NEAR) * 5))
    p_obs = [p_obs_sim, 1 - p_obs_sim]

    if abs(dist) < DIST_NEAR:
        if dist < 0:  # obstáculo à esquerda
            if angle < -ANGLE_FRONT:
                p_dir = [0.2, 0.1, 0.7]
            elif angle > ANGLE_FRONT:
                p_dir = [0.3, 0.1, 0.6]
            else:
                p_dir = [0.0, 0.3, 0.7]
        else:  # obstáculo à direita
            if angle > ANGLE_FRONT:
                p_dir = [0.7, 0.1, 0.2]
            elif angle < -ANGLE_FRONT:
                p_dir = [0.6, 0.1, 0.3]
            else:
                p_dir = [0.7, 0.3, 0.0]
    else:
        if angle < -ANGLE_FRONT:
            p_dir = [0.8, 0.1, 0.1]
        elif angle > ANGLE_FRONT:
            p_dir = [0.1, 0.1, 0.8]
        else:
            p_dir = [0.1, 0.8, 0.1]

    if VISION:
        p_vis_sim = probTargetVisible(
            camera.getImageArray())
    else:
        # 45 degrees in radians
        p_vis_sim = 1.0 if abs(angle) < 0.7854 else 0.1
    p_vis = [p_vis_sim, 1 - p_vis_sim]

    # Virtual evidence: lista de fatores (um para cada variável)
    virtual_evidence = [
        TabularCPD('ObstacleDetected', 2, [[p_obs[0]], [p_obs[1]]],
                   state_names={'ObstacleDetected': ['sim', 'nao']}),
        TabularCPD('TargetVisible', 2, [[p_vis[0]], [p_vis[1]]],
                   state_names={'TargetVisible': ['sim', 'nao']}),
        TabularCPD('Direction', 3, [[p_dir[0]], [p_dir[1]], [p_dir[2]]],
                   state_names={'Direction': ['esquerda', 'frente', 'direita']})
    ]
    for ev in virtual_evidence:
        print(ev)

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
