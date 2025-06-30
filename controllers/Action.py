import math
# Definindo as funções que as ações devem chamar
MAX_SPEED = 6.28  # maximum speed of the robot's motors
TURN_RATE_FOWARD = 0.3
TURN_RATE_BACKWARD = 0.1
SPEED = 0.4
SMOOTH = "normal"
velocity = [0.0, 0.0, 0.0, 0.0]


def continueAction(timestep=0):
    """Função para a ação 'seguir'."""
    print("Executando: SEGUIR (continuar em frente)")
    velocity[0] = MAX_SPEED * SPEED
    velocity[1] = MAX_SPEED * SPEED
    velocity[2] = MAX_SPEED * SPEED
    velocity[3] = MAX_SPEED * SPEED


def turnLeftAction(timestep=0):
    """Função para a ação 'virar esquerda'."""
    print("Executando: VIRAR ESQUERDA")
    velocity[0] = MAX_SPEED * TURN_RATE_BACKWARD
    velocity[1] = MAX_SPEED * TURN_RATE_FOWARD
    velocity[2] = MAX_SPEED * TURN_RATE_BACKWARD
    velocity[3] = MAX_SPEED * TURN_RATE_FOWARD


def turnRightAction(timestep=0):
    """Função para a ação 'virar direita'."""
    print("Executando: VIRAR DIREITA")
    velocity[0] = MAX_SPEED * TURN_RATE_FOWARD
    velocity[1] = MAX_SPEED * TURN_RATE_BACKWARD
    velocity[2] = MAX_SPEED * TURN_RATE_FOWARD
    velocity[3] = MAX_SPEED * TURN_RATE_BACKWARD


def stopAction(timestep=0):
    """Função para a ação 'parar'."""
    print("Executando: PARAR")
    velocity[0] = (0)
    velocity[1] = (0)
    velocity[2] = (0)
    velocity[3] = (0)


def updateWheels(wheels, velocity):
    for i in range(4):
        if SMOOTH == "exp":
            current_vel = wheels[i].getVelocity()
            target_vel = velocity[i]
            smoothing = 0.2
            # exponential smoothing
            new_vel = current_vel * (1 - smoothing) + \
                target_vel * (1 - (1 - smoothing) ** 2)
            wheels[i].setVelocity(new_vel)
        else:
            wheels[i].setVelocity(velocity[i])


# O dicionário que mapeia as strings de ação para as funções
action_map = {
    "seguir": continueAction,
    "v_esq": turnLeftAction,
    "v_dir": turnRightAction,
    "parar": stopAction,
}
