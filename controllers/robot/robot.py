from controller import Robot, Motor, Lidar, Camera, Supervisor # type: ignore
import numpy as np

import Action
from Infer import bayesian,GPS


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

step_count = 0
max_steps = 1000

while robot.step(timestep) != -1 and step_count < max_steps:
    # Movimento automático simples: frente + curva suave
    for i in range(2):
        wheels[i].setVelocity(3.0)

    #TAREFA
    # Coleta de dados com a CNN
    # lidar_data = lidar.getRangeImage() #[0.1,0.2, 3.0 ....]
    # camera_data = camera.getImageArray() # (shape camera_w, camera_h, 3) 
    # DistToObject, AngToTarget = CNN(lidar_data,camera_data)

    # remover essa linha abaixo quando tiver o CNN
    DistToObject, AngToTarget = GPS(robot_node, obstacle_nodes)
    print("Distancia Objeto Mais Próximo",DistToObject)
    print("Distancia Bolinha AMarela", AngToTarget)
    action, p_sucess = bayesian(DistToObject,AngToTarget)

    Action.action_map[action](wheels[0], wheels[1],wheels[2],wheels[3])
    if p_sucess >= 0.9: break
    
    
    # np.savez(os.path.join(SAVE_PATH, f"sample_{step_count}.npz"),
    #          lidar=np.array(lidar_data),
    #          camera=np.array(camera_data),
    #          dist=min_dist,
    #          ang=min_angle)
    
    step_count += 1


print("Robô chegou ao seu destino.")
