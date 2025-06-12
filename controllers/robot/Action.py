import math

# Definindo as funções que as ações devem chamar
MAX_SPEED = 6.28  # maximum speed of the robot's motors

def continueAction(left,right,timestep = 0):
    """Função para a ação 'seguir'."""
    print("Executando: SEGUIR (continuar em frente)")
    left.setVelocity(MAX_SPEED * 0.25)
    right.setVelocity(MAX_SPEED * 0.25)

def turnLeftAction(left,right,turn_rate,timestep = 0):
    """Função para a ação 'virar esquerda'."""
    print("Executando: VIRAR ESQUERDA")
    left.setVelocity(MAX_SPEED * -0.1)
    right.setVelocity(MAX_SPEED * 0.3)


def turnRightAction(left,right,turn_rate,timestep = 0):
    """Função para a ação 'virar direita'."""
    print("Executando: VIRAR DIREITA")
    left.setVelocity(MAX_SPEED * 0.3)
    right.setVelocity(MAX_SPEED * -0.1)

def stopAction(left,right,timestep = 0):
    """Função para a ação 'parar'."""
    print("Executando: PARAR")
    left.setVelocity(0)
    right.setVelocity(0)

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

