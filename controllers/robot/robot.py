#!/home/dino/Documents/ia/.venv/bin/python3
from controller import Robot, Motor, Lidar, Camera, Supervisor  # type: ignore

from Infer import bayesian, mapSoftEvidence
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import Action as Action

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Motores
wheels= [
    robot.getDevice('back left wheel'),
    robot.getDevice('back right wheel'),
    robot.getDevice('front left wheel'),
    robot.getDevice('front right wheel')
]
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)


# Sensores
lidar = robot.getDevice('Ibeo Lux')
lidar.enable(timestep)
lidar.enablePointCloud()
camera = robot.getDevice('camera')
camera.enable(timestep)

# Supervisor para acessar posição do robô e obstáculos
robot_node = robot.getSelf()

TARGET = robot.getFromDef("TARGET")

while robot.step(timestep) != -1:

    # TAREFA
    # Coleta de dados com a CNN
    # lidar_data = lidar.getRangeImage() #[0.1,0.2, 3.0 ....]
    # camera_data = camera.getImageArray() # (shape camera_w, camera_h, 3)
    # DistToObject, AngToTarget = CNN(lidar_data,camera_data)
    # remover essa linha abaixo quando tiver o CNN

    soft_evidence = mapSoftEvidence(robot_node, lidar,camera,TARGET)
    action, p_sucess = bayesian(soft_evidence=soft_evidence)

    if p_sucess >= 0.9:
        break

    Action.action_map[action]()
    Action.updateWheels(wheels, Action.velocity)
    # np.savez(os.path.join(SAVE_PATH, f"sample_{step_count}.npz"),
    #          lidar=np.array(lidar_data),
    #          camera=np.array(camera_data),
    #          dist=min_dist,
    #          ang=min_angle)

print("Robô chegou ao seu destino.")
