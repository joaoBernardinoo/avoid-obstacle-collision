import math

# Definindo as funções que as ações devem chamar
MAX_SPEED = 6.28  # maximum speed of the robot's motors

def continueAction(b_left,b_right,f_left,f_right,timestep = 0):
    """Função para a ação 'seguir'."""
    print("Executando: SEGUIR (continuar em frente)")
    b_left.setVelocity(MAX_SPEED * 0.2)
    b_right.setVelocity(MAX_SPEED * 0.2)
    f_left.setVelocity(MAX_SPEED * 0.2)
    f_right.setVelocity(MAX_SPEED * 0.2)    

def turnLeftAction(b_left,b_right,f_left,f_right,turn_rate=0,timestep = 0):
    """Função para a ação 'virar esquerda'."""
    print("Executando: VIRAR ESQUERDA")
    b_left.setVelocity(MAX_SPEED * -0.3)
    f_left.setVelocity(MAX_SPEED * -0.3)
    b_right.setVelocity(MAX_SPEED * 0.3)
    f_right.setVelocity(MAX_SPEED * 0.3)


def turnRightAction(b_left,b_right,f_left,f_right,turn_rate=0,timestep = 0):
    """Função para a ação 'virar direita'."""
    print("Executando: VIRAR DIREITA")
    b_left.setVelocity(MAX_SPEED * 0.3)
    f_left.setVelocity(MAX_SPEED * 0.3)
    b_right.setVelocity(MAX_SPEED * -0.3)
    f_right.setVelocity(MAX_SPEED * -0.3)

def stopAction(b_left,b_right,f_left,f_right,timestep = 0):
    """Função para a ação 'parar'."""
    print("Executando: PARAR")
    b_left.setVelocity(0)
    b_right.setVelocity(0)
    f_left.setVelocity(0)
    f_right.setVelocity(0)


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

# O dicionário que mapeia as strings de ação para as funções
action_map = {
    "seguir": continueAction,
    "v_esq": turnLeftAction,
    "v_dir": turnRightAction,
    "parar": stopAction,
}

