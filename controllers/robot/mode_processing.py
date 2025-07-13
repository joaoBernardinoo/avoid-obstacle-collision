import numpy as np
import random
from sensor_processing import GPS
from data_collection import collectDataHDF5
import sys
from pathlib import Path

from neural import MLP

if True:
    sys.path.append(str(Path(__file__).parent.parent))
    from cnn.NeuralNetwork import CNN


def _process_gps_mode(robot_node, lidar_data, target):
    dist, angle = GPS(robot_node, lidar_data, target)
    print("GPS - dist", dist)
    print("GPS - angle", angle * 180)
    angle = np.multiply(angle, np.pi)
    return dist, angle, False


def _process_nav_mode(lidar_data, camera_data, target):
    """
    Processa dados de Lidar e Câmera usando uma rede neural para prever distância e ângulo.

    Args:
        lidar_data (List[float]): Array de 20 floats com os dados do LiDAR.
        camera_data (bytes): Imagem da câmera no formato BGR (H, W, 3).

    Returns:
        tuple[float, float, bool]: Uma tupla contendo a distância e o ângulo previstos.
    """
    # camera_data_np = np.frombuffer(
    #     camera_data, np.uint8
    # ).reshape((40, 200, 4))
    dist = MLP(lidar_data)
    dist_gps, AngToTarget = GPS(lidar_data, camera_data, target)
    print("MLP - dist", dist)
    print("GPS - dist", dist_gps)
    print("CNN - angle", AngToTarget * 180)
    AngToTarget = np.multiply(AngToTarget, np.pi)
    return dist, AngToTarget, False


def _process_online_mode(robot_node, lidar_data, camera, target, camera_data):
    reset = False
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
    return dist, angle, reset


def _process_train_mode(robot_node, lidar_data, camera, target, camera_data):
    reset = False
    dist_gps, angle_gps = GPS(robot_node, lidar_data, target)
    if abs(dist_gps) <= 0.21:
        reset = True

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    # collectData(dist_gps, angle_gps, lidar_data, camera_data)
    collectDataHDF5(dist_gps, angle_gps, lidar_data,
                    camera_data, cam_w, cam_h)

    dist_ = dist_gps
    angle = angle_gps
    return dist_, angle, reset


def process_mode(mode, robot_node, lidar, camera, target, lidar_data, camera_data):
    if mode == "nav":
        return _process_nav_mode(lidar_data, camera_data,target)
    elif mode == "online":
        return _process_online_mode(robot_node, lidar_data, camera, target, camera_data)
    elif mode == "train":
        return _process_train_mode(robot_node, lidar_data, camera, target, camera_data)
    elif mode == "gps":
        return _process_gps_mode(robot_node, lidar_data, target)
    else:
        raise ValueError(f"Unknown MODE: {mode}")
