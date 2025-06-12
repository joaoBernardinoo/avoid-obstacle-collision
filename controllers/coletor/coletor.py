"""
Script para o robô Pioneer andar de forma automática no Webots e coletar dados para treinamento.
Salva: imagem da câmera, dados do LIDAR, distância e ângulo até o obstáculo mais próximo.
"""
from controller import Robot, Motor, Lidar, Camera, Supervisor # type: ignore
import numpy as np
import os
import math

SAVE_PATH = "../dataset"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Motores
wheels = [
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
# Obstáculos e objetivo conforme nomes do .wbt
find_nodes = [
    'box(1)',
    'box(2)',
    'box(3)',
    'blue-ball',
    'blue-ball(1)',
    'blue-ball(2)',
    'blue-ball(3)',
    'blue-ball(4)',
    'blue-ball(5)',
    'objective']

root_node = robot.getRoot()
children_field = root_node.getField('children')
num_children = children_field.getCount()

obstacle_nodes = []

for i in range(num_children):
    child_node = children_field.getMFNode(i)
    name_field = child_node.getField('name') # Get the 'name' field object
    if name_field:
        node_name = name_field.getSFString() # Get the string value of the 'name' field
        if node_name in find_nodes:
            obstacle_nodes.append(child_node)
            break

def getClosestObstacle(robot_node, obstacle_nodes):
    # Posição do robô
    rob_pos = robot_node.getPosition()
    rob_rot = robot_node.getOrientation()
    rob_angle = math.atan2(rob_rot[1], rob_rot[0])

    # Posição do obstáculo mais próximo
    min_dist = float('inf')
    min_angle = 0.0
    for obs in obstacle_nodes:
        obs_pos = obs.getPosition()
        dx = obs_pos[0] - rob_pos[0]
        dy = obs_pos[1] - rob_pos[1]
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) - rob_angle
        if dist < min_dist:
            min_dist = dist
            min_angle = angle
    
    return min_dist, min_angle
        
step_count = 0
max_steps = 1000

while robot.step(timestep) != -1 and step_count < max_steps:
    # Movimento automático simples: frente + curva suave
    for i, wheel in enumerate(wheels):
        if i % 2 == 0:
            wheel.setVelocity(2.0)
        else:
            wheel.setVelocity(1.5)

    # Coleta de dados
    lidar_data = lidar.getRangeImage()
    camera_data = camera.getImageArray()

    dist, ang = getClosestObstacle(robot_node, obstacle_nodes)


    # np.savez(os.path.join(SAVE_PATH, f"sample_{step_count}.npz"),
    #          lidar=np.array(lidar_data),
    #          camera=np.array(camera_data),
    #          dist=min_dist,
    #          ang=min_angle)
    
    step_count += 1



print("Coleta finalizada.")
