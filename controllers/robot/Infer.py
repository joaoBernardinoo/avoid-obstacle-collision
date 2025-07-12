import numpy as np
import logging
import sys
from pathlib import Path
from pgmpy.factors.discrete import TabularCPD
import random
import Bayes
from typing import List

from .constants import VISION, DIST_NEAR, MODE, ANGLE_FRONT
from .sensor_processing import probTargetVisible, GPS
from .data_collection import collectDataHDF5

if True:
    sys.path.append(str(Path(__file__).parent.parent))
    from cnn.NeuralNetwork import CNN, CNN_online

logging.getLogger("pgmpy").setLevel(logging.ERROR)

inference = Bayes.load_model()


def mapSoftEvidence(robot_node, lidar, camera, target):

    reset = False

    # 1. Obter dados dos sensores
    lidar_data = lidar.getRangeImage()  # type: List[int]
    camera_data = camera.getImage()    # Retorna uma string de bytes

    if MODE == "nav":
        camera_data_np = np.frombuffer(
            camera_data, np.uint8
        ).reshape((40, 200, 4))
        dist, angle = CNN(lidar_data, camera_data_np)

        dist = np.multiply(dist, 3.14)
        print("CNN - dist", dist)

        print("CNN - angle", angle * 180)
        angle = np.multiply(angle, np.pi)

    elif MODE == "online":
        camera_data_np = np.frombuffer(
            camera_data, np.uint8
        ).reshape((40, 200, 4))
        dist, angle = CNN(lidar_data, camera_data_np)
        dist = np.multiply(dist, 3.14)
        angle = np.multiply(angle, np.pi)

        dist2, angle2 = GPS(robot_node, lidar_data, target)
        # Check if the difference between CNN output and ground truth is greater than 30%
        dist_diff = abs(dist - dist2)
        angle_diff = abs(angle - angle2)

        dist_threshold = 0.3 * abs(dist2) if dist2 != 0 else 0.30
        angle_threshold = 0.3 * abs(angle2) if angle2 != 0 else 0.30

        print("CNN - dist", dist)
        print("Ground Truth", dist2)
        print("CNN - angle", angle * 180 / np.pi)
        print("Ground Truth", angle2 * 180 / np.pi)
        if abs(dist2) <= 0.21:
            reset = True

        if dist_diff > dist_threshold or angle_diff > angle_threshold:
            print(
                "CNN output differs significantly from ground truth. Calling CNN_online...")
            # CNN_online(lidar_data, camera_data_np, dist2, angle2)
            dist = dist2
            angle = angle2
    elif MODE == "train":
        dist_gps, angle_gps = GPS(robot_node, lidar_data, target)
        if abs(dist_gps) <= 0.21:
            reset = True

        choice = random.random() < 0.40
        if choice:
            camera_data_np = np.frombuffer(
                camera_data, np.uint8
            ).reshape((40, 200, 4))
            dist_cnn, angle_cnn = CNN(lidar_data, camera_data_np)
            dist_cnn = dist_cnn * np.pi
            angle_cnn = angle_cnn * np.pi

            # print the cnn values and the gps values
            print("CNN - dist", dist_cnn)
            print("Ground Truth", dist_gps)
            print("CNN - angle", angle_cnn * 180 / np.pi)
            print("Ground Truth", angle_gps * 180 / np.pi)
            dist = dist_cnn
            angle = angle_cnn

        else:
            dist = dist_gps
            angle = angle_gps

        cam_w = camera.getWidth()
        cam_h = camera.getHeight()
        # collectData(dist_gps, angle_gps, lidar_data, camera_data)
        collectDataHDF5(dist_gps, angle_gps, lidar_data, camera_data, cam_w, cam_h )

    if VISION:
        p_vis_sim = probTargetVisible(
            camera_data)
    else:
        # 45 degrees in radians
        p_vis_sim = 1.0 if abs(angle) < 0.7854 else 0.1
    p_vis = [p_vis_sim, 1 - p_vis_sim]

    # Soft evidence (probabilidades)
    p_obs_sim = 1 / (1 + np.exp((abs(dist) - DIST_NEAR) * 5))
    p_obs = [p_obs_sim, 1 - p_obs_sim]

    # --- Probabilidade da Direção (p_dir) ---
    # Esta seção mapeia a probabilidade da direção com base no ângulo para o alvo
    # e na probabilidade de detecção de obstáculo (p_obs_sim),
    # tornando a transição de comportamento mais suave.

    # 1. Definir a probabilidade de direção baseada apenas no ângulo para o alvo.
    #    (Comportamento quando não há obstáculos)
    if angle < -ANGLE_FRONT:  # Alvo à esquerda
        p_dir_target = np.array([0.8, 0.1, 0.1])
    elif angle > ANGLE_FRONT:  # Alvo à direita
        p_dir_target = np.array([0.1, 0.1, 0.8])
    else:  # Alvo em frente
        p_dir_target = np.array([0.1, 0.8, 0.1])

    # 2. Definir a probabilidade de direção para desviar de um obstáculo.
    #    (Comportamento quando um obstáculo está muito próximo)
    if dist < 0:  # Obstáculo detectado à esquerda
        p_dir_avoidance = np.array([0.1, 0.2, 0.7])  # Desviar para a direita
    else:  # Obstáculo detectado à direita ou em frente
        p_dir_avoidance = np.array([0.7, 0.2, 0.1])  # Desviar para a esquerda

    # 3. Misturar as duas probabilidades usando p_obs_sim como peso.
    #    Se p_obs_sim é alto, o robô prioriza desviar (p_dir_avoidance).
    #    Se p_obs_sim é baixo, o robô prioriza seguir o alvo (p_dir_target).
    # Aumentamos o peso de p_dir_target por um fator de 2 para que o robô
    # priorize mais fortemente o alvo quando não há obstáculos.
    p_dir = p_dir_target * p_obs[1] + p_dir_avoidance * p_obs[0]

    # Como a ponderação desbalanceada faz com que a soma das probabilidades não seja 1,
    # normalizamos o vetor para garantir que ele seja uma distribuição de probabilidade válida.
    p_dir /= np.sum(p_dir)

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

    return virtual_evidence, reset


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
