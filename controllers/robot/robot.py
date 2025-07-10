#!/home/dino/Documents/ia/.venv/bin/python3
from controller import Robot, Motor, Lidar, Camera, Supervisor  # type: ignore
import cv2
import numpy as np

from Infer import bayesian, mapSoftEvidence
import sys
from pathlib import Path
from typing import List

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
    # only infer the action in steps multiples of 5
    soft_evidence, reset = mapSoftEvidence(robot_node, lidar, camera, TARGET)
    action, p_sucess = bayesian(soft_evidence=soft_evidence)
    if p_sucess >= 0.9:
        break
    if reset or action == "parar":
        # reset the robot to the start position
        # translation -1.89737 1.92596 -0.081334
        # rotation -0.011571497788369405 -0.016505796845289522 -0.9997968089114087 0.869511
        robot_node.getField('translation').setSFVec3f(
            [-1.89737, 1.92596, -0.081334])
        robot_node.getField('rotation').setSFRotation(
            [-0.011571497788369405, -0.016505796845289522, -0.9997968089114087, 0.869511])

    if cv2.waitKey(1) == ord('q'):
        break

    Action.action_map[action]()
    Action.updateWheels(wheels, Action.velocity)
    # np.savez(os.path.join(SAVE_PATH, f"sample_{step_count}.npz"),
    #          lidar=np.array(lidar_data),
    #          camera=np.array(camera_data),
    #          dist=min_dist,
    #          ang=min_angle)

print("Robô chegou ao seu destino.")
