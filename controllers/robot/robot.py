#!/home/dino/Documents/avoid-obstacle-collision/.venv/bin/python
from controller import Motor, Lidar, Camera, Supervisor  # type: ignore
import cv2
from mode_processing import process_mode
from Infer import bayesian, mapSoftEvidence
from constants import MODE
import sys
import numpy as np
from pathlib import Path
from typing import List
sys.path.append(str(Path(__file__).parent))

if True:
    sys.path.append(str(Path(__file__).parent.parent))
    import Action as Action


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Motores
wheels: List[Motor] = [  # type: ignore
    robot.getDevice('back left wheel'),
    robot.getDevice('back right wheel'),
    robot.getDevice('front left wheel'),
    robot.getDevice('front right wheel')
]
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)


# Sensores
lidar: Lidar = robot.getDevice('Ibeo Lux')  # type: ignore
lidar.enable(timestep)
lidar.enablePointCloud()
camera: Camera = robot.getDevice('camera')  # type: ignore
camera.enable(timestep)


# Supervisor para acessar posição do robô e obstáculos
robot_node = robot.getSelf()

# put in NW corner
cv2.namedWindow("Webots Camera")
cv2.moveWindow("Webots Camera", 0, 2)

TARGET = robot.getFromDef("TARGET")
step_count = 0
while robot.step(timestep) != -1:
    lidar_data = lidar.getRangeImage()  # type: List[float]
    camera_data = camera.getImage()    # Retorna uma string de bytes
    image = np.frombuffer(camera_data, np.uint8).reshape((40, 200, 4))

    dist, angle, reset = process_mode(
        MODE, robot_node, lidar, camera, TARGET, lidar_data, image)

    soft_evidence = mapSoftEvidence(dist, angle, image)
    action, p_sucess = bayesian(soft_evidence=soft_evidence)

    if cv2.waitKey(1) == ord('q'):
        break

    Action.action_map[action]()
    Action.updateWheels(wheels, Action.velocity)

    if p_sucess >= 0.9:
        wheels[0].setVelocity(0.0)
        wheels[1].setVelocity(0.0)
        wheels[2].setVelocity(0.0)
        wheels[3].setVelocity(0.0)
        wheels[0].setVelocity(0.0)
        break
    if reset:
        wheels[0].setVelocity(0.0)
        wheels[1].setVelocity(0.0)
        wheels[2].setVelocity(0.0)
        wheels[3].setVelocity(0.0)
        wheels[0].setVelocity(0.0)
        robot_node.getField('translation').setSFVec3f(
            [-1.89737, 1.92596, -0.081334])
        robot_node.getField('rotation').setSFRotation(
            [-0.011571497788369405, -0.016505796845289522, -0.9997968089114087, 0.869511])
print("Robô chegou ao seu destino.")
