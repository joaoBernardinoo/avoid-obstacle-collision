import numpy as np
import logging
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os
import cv2
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
        [0.10, 0.55, 0.10, 0.25],  # ( sim, nao, esquerda )
        [0.10, 0.35, 0.35, 0.20],  # ( sim, nao, frente )
        [0.10, 0.10, 0.55, 0.25],  # ( sim, nao, direita )
        [0.70, 0.15, 0.15, 0.00],  # ( nao, sim, esquerda )
        [0.60, 0.20, 0.20, 0.00],  # ( nao, sim, frente )
        [0.10, 0.10, 0.80, 0.00],  # ( nao, sim, direita )
        [0.10, 0.50, 0.30, 0.10],  # ( nao, nao, esquerda )
        [0.10, 0.40, 0.40, 0.10],  # ( nao, nao, frente )
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


def probTargetVisible(image) -> float:
    # Converter para HSV
    hsv = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_BGR2HSV)

    # Definir faixa de amarelo (ajuste conforme o ambiente!)
    lower_yellow = np.array([62, 44, 14])
    upper_yellow = np.array([242, 210, 30])

    # Máscara para regiões amarelas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

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
    return angle


REPULSE = "cos"
VISION = True


def GPS(robot_node, lidar, target):
    # Posição do robô
    """
    Calculate the minimum distance to the nearest obstacle and the angle to the target.

    This function uses LIDAR data to determine the closest distance to any obstacle
    and calculates the angle to a specific target, identified as 'objective', from
    the robot's current position.

    Args:
        robot_node: The node representing the robot in the simulator.
        lidar: The LIDAR device used to scan the environment.

    Returns:
        A tuple containing:
        - min_dist (float): The distance to the nearest obstacle.
        - angle (float): The angle to the target, 'objective', in radians.
    """

    lidar_data = lidar.getRangeImage()  # type: List[int]
    # Posição do obstáculo mais próximo
    min_dist = min(lidar_data)
    angle = 0.0
    angle = getAngle(robot_node, target)

    return min_dist, -angle


def mapSoftEvidence(robot_node, lidar, camera, target):

    dist, angle = GPS(robot_node, lidar, target)
    print("Angulo Bolinha Amarela", angle * 180 / np.pi)
    print("Distancia Objeto Mais Próximo", dist)
    DIST_NEAR = 0.8
    DIST_STOP = 0.05
    ANGLE_FRONT = 0.0218  # 15 degrees in radians

    # Soft evidence (probabilidades)
    p_obs_sim = 1 / (1 + np.exp((dist - DIST_NEAR) * 5))
    p_obs = [p_obs_sim, 1 - p_obs_sim]

    # Deve-se encontrar a probabilidade de visibilidade do alvo
    # com base na camera e no angulo

    if DIST_NEAR > dist:
        # multiply the angle proportional to the distance
        # the angle is inverse proportional to the distance
        min_index = lidar.getRangeImage().index(dist)
        fov = lidar.getFov()
        # Calculate the angle in radians corresponding to the index of the minimum distance
        # Assuming lidar_data covers a field of view of 180 degrees (π radians)
        fov2 = fov/2
        angle = (min_index / lidar.getHorizontalResolution()) * fov - fov2
        # normalize to [-fov2,fov2]
        # increse the angle based on the distance
        # the angle is inverse proportional to the distance
        # because when a obstacle is closer whe should:
        # - mirror the angle of the obstacle, because we should go in the opposite direction to avoid
        # - increase the angle based on the distance, cause the closer the obstacle the more we should turn
        if REPULSE == "cos":
            b = 3.0
            c = 2
            # lidar
            avoid_multiplier = np.cos(angle/c) ** 2
            dist_multiplier = -math.log(DIST_NEAR / (dist + 0.01))
            angle_multiplier = avoid_multiplier * dist_multiplier
        elif REPULSE == "exp":
            angle_multiplier = (1/np.exp((angle / fov2)**2)) * - \
                math.log(DIST_NEAR / (dist + 0.01))
        else:
            angle_multiplier = -math.log(DIST_NEAR / (dist + 0.01))

        print("Angle multiplier", angle_multiplier)
        angle *= angle_multiplier

    if VISION:
        p_vis_sim = probTargetVisible(
            camera.getImageArray())
    else:
        # 45 degrees in radians
        p_vis_sim = 1.0 if abs(angle) < 0.7854 else 0.1
    p_vis = [p_vis_sim, 1 - p_vis_sim]

    print("Angulo Mentiroso", angle * 180 / np.pi)

    if angle < -ANGLE_FRONT:
        p_dir = [0.8, 0.1, 0.1]
    elif angle > ANGLE_FRONT:
        p_dir = [0.1, 0.1, 0.8]
    else:
        p_dir = [0.1, 0.8, 0.1]

    # Virtual evidence: lista de fatores (um para cada variável)
    virtual_evidence = [
        TabularCPD('ObstacleDetected', 2, [[p_obs[0]], [p_obs[1]]],
                   state_names={'ObstacleDetected': ['sim', 'nao']}),
        TabularCPD('TargetVisible', 2, [[p_vis[0]], [p_vis[1]]],
                   state_names={'TargetVisible': ['sim', 'nao']}),
        TabularCPD('Direction', 3, [[p_dir[0]], [p_dir[1]], [p_dir[2]]],
                   state_names={'Direction': ['esquerda', 'frente', 'direita']})
    ]

    # debug
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
