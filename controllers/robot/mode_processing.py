import numpy as np
import random
from sensor_processing import GPS
from data_collection import collectDataHDF5
import sys
from pathlib import Path
import random
from neural import MLP

if True:
    sys.path.append(str(Path(__file__).parent.parent))
    from cnn.NeuralNetwork import CNN


def _process_gps_mode(robot_node, lidar_data, image, target):
    dist, angle = GPS(robot_node, lidar_data, target)
    AngToTarget = CNN(image)

    print("GPS - dist", dist)
    print("GPS - angle", angle * 180/np.pi)
    print("CNN - angle", AngToTarget * 180 / np.pi)
    angle = np.multiply(angle, np.pi)
    return dist, angle, False


def _process_nav_mode(robot_node, lidar_data, camera_data, target):
    """
    Processa dados de Lidar e Câmera usando uma rede neural para prever distância e ângulo.

    Args:
        lidar_data (List[float]): Array de 20 floats com os dados do LiDAR.
        camera_data (bytes): Imagem da câmera no formato BGR (H, W, 3).

    Returns:
        tuple[float, float, bool]: Uma tupla contendo a distância e o ângulo previstos.
    """
    image = np.frombuffer(
        camera_data, np.uint8
    ).reshape((40, 200, 4))
    DistToObject = MLP(lidar_data)
    AngToTarget = CNN(image)
    dist_gps, ang_gps = GPS(robot_node, lidar_data, target)
    print("MLP - DistToObject", DistToObject)
    print("GPS - dist", dist_gps)
    print("GPS - angle", ang_gps * 180 / np.pi)
    print("CNN - angle", AngToTarget * 180 / np.pi)
    return DistToObject, AngToTarget, False


def _process_collect_mode(robot_node, lidar_data, camera, target, camera_data):
    reset = False
    dist_gps, angle_gps = GPS(robot_node, lidar_data, target)
    if abs(dist_gps) <= 0.21:
        reset = True

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    # collectData(dist_gps, angle_gps, lidar_data, camera_data)
    collectDataHDF5(angle_gps,
                    camera_data, cam_w, cam_h)
    # add a randon noise of 1.5% of the distance
    angle = angle_gps
    dist = dist_gps
    print("GPS - dist", dist)
    print("GPS - angle", angle * 180 / np.pi)
    return dist, angle, reset


def _process_train_mode(robot_node, lidar_data, camera, target, camera_data):
    reset = False
    dist_gps, angle_gps = GPS(robot_node, lidar_data, target)
    if abs(dist_gps) <= 0.21:
        reset = True

    cam_w = camera.getWidth()
    cam_h = camera.getHeight()
    # collectData(dist_gps, angle_gps, lidar_data, camera_data)
    print("Collecting data...")
    collectDataHDF5(angle_gps,
                    camera_data, cam_w, cam_h)
    # add a randon noise of 1.5% of the distance
    dist = dist_gps
    angle = angle_gps
    print("GPS - dist", dist)
    print("GPS - angle", angle * 180 / np.pi)
    return dist, angle, reset


def process_mode(mode, robot_node, lidar, camera, target, lidar_data, camera_data):
    if mode == "nav":
        return _process_nav_mode(robot_node, lidar_data, camera_data, target)
    elif mode == "train":
        return _process_train_mode(robot_node, lidar_data, camera, target, camera_data)
    elif mode == "gps":
        return _process_gps_mode(robot_node, lidar_data, camera_data, target)
    elif mode == "collect":
        return _process_collect_mode(robot_node, lidar_data, camera, target, camera_data)
    else:
        raise ValueError(f"Unknown MODE: {mode}")
