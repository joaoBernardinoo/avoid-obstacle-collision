import numpy as np
import logging
import sys
from pathlib import Path
from pgmpy.factors.discrete import TabularCPD
import os
import cv2
import math
import random
import Bayes
from typing import List, cast
import h5py
if True:
    sys.path.append(str(Path(__file__).parent.parent))
    from cnn.NeuralNetwork import CNN, CNN_online

logging.getLogger("pgmpy").setLevel(logging.ERROR)

STEP_COUNT = 0
if sys.platform == 'linux':
    SAVE_PATH = Path('/home/dino/SSD/cnn_dataset')

else:
    SAVE_PATH = Path(os.path.dirname(__file__), 'cnn_dataset')

HDF5_SAVE_PATH = SAVE_PATH.parent / 'cnn_dataset.h5'


def get_limits(color_bgr: List[int]):
    """
    Calcula os limites de cor HSV inferior e superior a partir de uma cor BGR de entrada.
    Esta função foi adaptada da imagem que você forneceu.
    """
    # Criamos um array 3D para a conversão, como o cvtColor espera
    c = np.uint8([[color_bgr]])  # type: ignore
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)  # type: ignore

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

    mask = cv2.resize(mask, None, fx=3, fy=3)
    cv2.imshow("Webots Camera", mask)
    # Calcular proporção
    yellow_ratio = np.sum(mask > 0) / mask.size
    yellow_ratio = float(yellow_ratio)

    print(f"Proporção de pixels amarelos: {yellow_ratio:.2%}")
    # Definir probabilidade com uma transição suave de 0 a 1
    prob = 1 - np.power(2, -25 * yellow_ratio)
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


def collectData(dist, angle, lidar_data, camera_data):
    global STEP_COUNT

    # --- Início da Coleta de Dados para a CNN ---
    # O código abaixo coleta os dados dos sensores e os salva junto com os
    # valores de "ground truth" (dist, angle) para treinar a CNN posteriormente.

    # Nota: lidar.getRangeImage() é usado para a varredura 1D do Lidar.

    # 2. Converter a imagem da câmera de bytes para um array numpy
    # A imagem do Webots é BGRA (4 canais).
    image_np = np.frombuffer(
        camera_data, np.uint8
    ).reshape((40, 200, 4))

    # 3. Salvar a amostra de treinamento em um arquivo .npz.
    np.savez_compressed(
        os.path.join(SAVE_PATH, f"sample_{STEP_COUNT}.npz"),
        camera_image=image_np,
        lidar_data=np.array(lidar_data),
        dist=dist,
        angle=angle
    )
    STEP_COUNT += 1
    # --- Fim da Coleta de Dados ---

    return 0


def collectDataHDF5(dist, angle, lidar_data, camera_data):
    """
    Collects and appends sensor data to a single HDF5 file.

    The function saves the camera image, LIDAR data, and ground truth labels
    (distance and angle) into datasets within an HDF5 file. If the file or
    datasets don't exist, they are created. Subsequent calls append new data
    to the existing datasets, making them grow over time.
    """
    global STEP_COUNT

    # 1. Convert camera image from bytes to a numpy array
    image_np = np.frombuffer(
        camera_data, np.uint8
    ).reshape((40, 200, 4))
    lidar_np = np.array(lidar_data)

    # 2. Open the HDF5 file in append mode
    with h5py.File(HDF5_SAVE_PATH, 'a') as hf:
        # 3. Check if datasets exist. If not, create them.
        if 'camera_image' not in hf:
            # Create datasets that can be resized
            hf.create_dataset('camera_image', data=[image_np],
                              compression="gzip", chunks=True,
                              maxshape=(None, 40, 200, 4))
            hf.create_dataset('lidar_data', data=[lidar_np],
                              compression="gzip", chunks=True,
                              maxshape=(None, len(lidar_np)))
            hf.create_dataset('dist', data=[dist],
                              compression="gzip", chunks=True,
                              maxshape=(None,))
            hf.create_dataset('angle', data=[angle],
                              compression="gzip", chunks=True,
                              maxshape=(None,))
        else:
            # 4. If datasets exist, resize and append new data
            # We use cast to inform Pylance about the correct type
            camera_dset = cast(h5py.Dataset, hf['camera_image'])
            camera_dset.resize((camera_dset.shape[0] + 1), axis=0)
            camera_dset[-1] = image_np

            lidar_dset = cast(h5py.Dataset, hf['lidar_data'])
            lidar_dset.resize((lidar_dset.shape[0] + 1), axis=0)
            lidar_dset[-1] = lidar_np

            dist_dset = cast(h5py.Dataset, hf['dist'])
            dist_dset.resize((dist_dset.shape[0] + 1), axis=0)
            dist_dset[-1] = dist

            angle_dset = cast(h5py.Dataset, hf['angle'])
            angle_dset.resize((angle_dset.shape[0] + 1), axis=0)
            angle_dset[-1] = angle

    STEP_COUNT += 1
    return 0


REPULSE = "cos"
VISION = True
DIST_NEAR = 0.6
MODE = "online"
ANGLE_FRONT = 0.0218  # 15 degrees in radians


def GPS(robot_node, lidar_data, target):
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
    # Posição do obstáculo mais próximo
    min_dist = min(lidar_data)
    min_index = lidar_data.index(min_dist)

    angle = 0.0
    angle = getAngle(robot_node, target)
    if min_dist < DIST_NEAR and min_index < len(lidar_data) // 2:
        # min_dist é a distancia do ojbeto mais proximo, eu quero que
        # objetos a esquerda estejam com distancia negativa e direita com distancia positiva
        # se ele estiver no centro deve ser 0
        min_dist = -min_dist

    # add a random noise
    # min_dist += random.uniform(-0.05, 0.05)
    # angle += random.uniform(-ANGLE_FRONT, ANGLE_FRONT)

    return min_dist, angle


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

        # collectData(dist_gps, angle_gps, lidar_data, camera_data)
        collectDataHDF5(dist_gps, angle_gps, lidar_data, camera_data)

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


inference = Bayes.load_model()


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
