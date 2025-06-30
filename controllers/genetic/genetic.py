import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import Action as Action

from controller import Robot, Motor, Lidar, Camera, Supervisor  # type: ignore


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
    lidar_data = lidar.getRangeImage() #[0.1,0.2, 3.0 ....]
    camera_data = camera.getImageArray() # (shape camera_w, camera_h, 3)

    action = "seguir" # deve-se inferir usando um modelo CNN

    Action.action_map[action]()
    Action.updateWheels(wheels, Action.velocity)
    # np.savez(os.path.join(SAVE_PATH, f"sample_{step_count}.npz"),
    #          lidar=np.array(lidar_data),
    #          camera=np.array(camera_data),
    #          dist=min_dist,
    #          ang=min_angle)

print("Robô chegou ao seu destino.")
