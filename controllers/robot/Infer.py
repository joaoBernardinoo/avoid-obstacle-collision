import numpy as np
import logging
import cProfile
import pstats
from line_profiler import LineProfiler
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os,math
import time
import pandas as pd

logging.getLogger("pgmpy").setLevel(logging.ERROR)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bayes_model.bif')

# Forçar a recriação do modelo com as novas probabilidades
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print("[INFO] Modelo antigo removido, criando novo modelo")

if not os.path.exists(MODEL_PATH):
    # 1. Definir a estrutura da Rede Bayesiana (nós e arestas/dependências)
    # A estrutura reflete as dependências causais que definimos antes.
    model = DiscreteBayesianNetwork([
        ('ObstacleDetected', 'Action'),
        ('TargetVisible', 'Action'),
        ('Direction', 'Action'),
        # O sucesso em chegar ao destino depende da sequência de ações
        ('Action', 'Success')
    ])

    # CPT para a Ação. Depende de ObstacleDetected, TargetVisible e Direction.
    # 2. Definir as Tabelas de Probabilidade Condicional (CPTs)
    # --- CPTs para os Nós Raiz (A CORREÇÃO ESTÁ AQUI) ---
    # Usamos um valor padrão uniforme. Eles serão sobrescritos pela evidência.
    cpt_O = TabularCPD(
        variable='ObstacleDetected',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'ObstacleDetected': ['sim', 'nao']}
    )

    cpt_V = TabularCPD(
        variable='TargetVisible',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'TargetVisible': ['sim', 'nao']}
    )

    cpt_D = TabularCPD(
        variable='Direction',
        variable_card=3,
        values=[[0.33], [0.34], [0.33]],
        state_names={'Direction': ['esquerda', 'frente', 'direita']}
    )

    # A ordem das evidências em 'evidence' é crucial e deve corresponder ao shape do array 'values'.
    cpt_A = TabularCPD(
        variable='Action',
        variable_card=4,  # 4 estados: seguir, v_esq, v_dir, parar
        values=[
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.70, 0.60, 0.10, 0.10, 0.10, 0.10],  # seguir
            [0.35, 0.35, 0.10, 0.35, 0.35, 0.10, 0.15, 0.20, 0.10, 0.30, 0.30, 0.30],  # v_esq
            [0.10, 0.35, 0.35, 0.10, 0.35, 0.35, 0.15, 0.20, 0.80, 0.50, 0.50, 0.50],  # v_dir
            [0.45, 0.20, 0.45, 0.45, 0.20, 0.45, 0.00, 0.00, 0.00, 0.10, 0.10, 0.10]   # parar
        ],
        evidence=['ObstacleDetected', 'TargetVisible', 'Direction'],
        evidence_card=[2, 2, 3],  # O(2), V(2), D(3)
        state_names={
            'Action': ['seguir', 'v_esq', 'v_dir', 'parar'],
            'ObstacleDetected': ['sim', 'nao'],
            'TargetVisible': ['sim', 'nao'],
            'Direction': ['esquerda', 'frente', 'direita']
        }
    )

    # CPT para o Sucesso. Por sua definição, ele depende da ação tomada.
    # A probabilidade de sucesso é a chance de eventualmente chegar no destino.
    cpt_S = TabularCPD(
        variable='Success',
        variable_card=2,  # 2 estados: sim, nao
        values=[
            [0.90,   0.60,   0.60,  0.99],  # Probabilidade de sucesso 'sim'
            [0.10,   0.40,   0.40,  0.01]  # Probabilidade de sucesso 'nao'
        ],
        evidence=['Action'],
        evidence_card=[4],
        state_names={
            'Success': ['sim', 'nao'],  # Standardized to 'nao' to match other CPDs
            'Action': ['seguir', 'v_esq', 'v_dir', 'parar']
        }
    )

    
    # Adicionar as CPTs ao modelo
    model.add_cpds(cpt_A, cpt_S,cpt_D,cpt_O,cpt_V)
    # Verificar se o modelo e as CPTs são consistentes
    if not model.check_model():
        raise ValueError("O modelo Bayesiano não é válido.")
    model.save(MODEL_PATH)
    print("[INFO] Modelo Bayesiano salvo em bayes_model")



# Criar um objeto de inferência
inference = VariableElimination(model)



## make cnn return the distance to objetct and ang to target
def CNN(lidar,camera) -> tuple[float, float]:

    return 5.0, 0.0  # Simula uma distância de 5 metros e ângulo de 0 radianos


def getAngle(robot_node, obstacle_node) -> float:
    """Calcula o ângulo entre o robô e o obstáculo."""
    obs_pos = obstacle_node.getPosition()
    rob_pos = robot_node.getPosition()
    rob_rot = robot_node.getOrientation()
    rob_yaw = math.atan2(rob_rot[1], rob_rot[0])
    dx = obs_pos[0] - rob_pos[0]
    dy = obs_pos[1] - rob_pos[1]
    return math.atan2(dy, dx) - rob_yaw

def GPS(robot_node, obstacle_nodes):
    # Posição do robô
    rob_pos = robot_node.getPosition()
    # Posição do obstáculo mais próximo
    min_dist = float('inf')
    angle = 0.0
    for obs in obstacle_nodes:
        if angle == 0.0 and obs.getField('name').getSFString() == 'objective':
            angle = getAngle(robot_node, obs)

        obs_pos = obs.getPosition()
        dx = obs_pos[0] - rob_pos[0]
        dy = obs_pos[1] - rob_pos[1]
        dist = math.hypot(dx, dy)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist, angle

# --- Função que usa o modelo pgmpy ---


def bayesian(dist: float, angle: float) -> tuple[str, float]:
    """Usa o modelo pgmpy para inferir a ação e a probabilidade de sucesso com soft evidence."""
    action_map = {0: 'seguir', 1: 'v_esq', 2: 'v_dir', 3: 'parar'}

    DIST_NEAR = 0.5
    DIST_STOP = 0.05
    ANGLE_FRONT = 0.2618  # 15 degrees in radians

    # Soft evidence (probabilidades)
    p_obs_sim = 1 / (1 + np.exp((dist - DIST_NEAR) * 5)) # 
    p_obs = [p_obs_sim, 1 - p_obs_sim]

    # Deve-se encontrar a probabilidade de visibilidade do alvo
    # com base na camera e no angulo
    p_vis_sim = 1.0 if abs(angle) < 0.7854 else 0.1  # 45 degrees in radians
    p_vis = [p_vis_sim, 1 - p_vis_sim]
    if angle < -ANGLE_FRONT:
        p_dir = [0.8, 0.1, 0.1]
    elif angle > ANGLE_FRONT:
        p_dir = [0.1, 0.1, 0.8]
    else:
        p_dir = [0.1, 0.8, 0.1]

    # Virtual evidence: lista de fatores (um para cada variável)
    virtual_evidence = [
        TabularCPD('ObstacleDetected', 2, [[p_obs[0]], [p_obs[1]]], 
                  state_names={'ObstacleDetected': ['sim', 'nao']}),
        TabularCPD('TargetVisible', 2, [[p_vis[0]], [p_vis[1]]],
                  state_names={'TargetVisible': ['sim', 'nao']}),
        TabularCPD('Direction', 3, [[p_dir[0]], [p_dir[1]], [p_dir[2]]],
                  state_names={'Direction': ['esquerda', 'frente', 'direita']})
    ]

    # debug
    # for ev in virtual_evidence:
    #     print(ev)

    # Inferir a distribuição de Action com soft evidence
    action_dist = inference.query(
        variables=['Action'], virtual_evidence=virtual_evidence) # type: ignore

    # debug
    print(action_dist)

    action_idx = int(np.argmax(action_dist.values)) # type: ignore
    action_str = action_map[action_idx]

    if dist < DIST_STOP and abs(angle) < ANGLE_FRONT:
        action_str = "parar"
        action_idx = 3

    # # Probabilidade de sucesso para a ação escolhida
    # prob_success_dist = inference.query(
    #     variables=['Success'],
    #     evidence={'Action': action_str}  # Use state name instead of index
    # ) # type: ignore
    # p_success = prob_success_dist.values[0] # type: ignore
    p_success = 0.0
    return action_str, p_success # type: ignore


def profile_bayesian():
    """Function to profile the bayesian inference with different scenarios"""
    scenarios = [
        (5.0, 0.0),        # Cenário 1 (Livre)
        (0.6, 0.0873),     # Cenário 2 (Desvio) - 5 degrees in radians
        (4.0, 0.7854),     # Cenário 3 (Correção) - 45 degrees in radians
        (0.2, 0.0349),     # Cenário 4 (Chegada) - 2 degrees in radians
    ]
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple iterations to get better data
    iterations = 300
    start_time = time.time()
    
    for _ in range(iterations):
        for dist, angle in scenarios:
            action, success = bayesian(dist, angle)
    
    end_time = time.time()
    profiler.disable()
    
    # Print timing statistics
    total_time = end_time - start_time
    avg_time = total_time / (iterations * len(scenarios))
    print(f"\nTiming Statistics:")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per inference: {avg_time*1000:.3f} ms")
    
    # Print detailed profiling statistics
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    print("\nDetailed Profile:")
    stats.print_stats(20)  # Print top 20 time-consuming functions
    
    # Line by line profiling of the bayesian function
    line_profiler = LineProfiler()
    lp_wrapper = line_profiler(bayesian)
    
    # Profile one scenario
    lp_wrapper(5.0, 0)
    
    print("\nLine by line profiling of bayesian():")
    line_profiler.print_stats()

if __name__ == '__main__':
    # Teste de cenários variados para comportamento do robô
    scenarios = [
        (5.0, 0.0, "Livre", "seguir"),          # Longe de obstáculos, alvo à frente
        (0.6, 0.0873, "Desvio", "v_esq/v_dir"), # Obstáculo próximo, leve desvio à direita - 5 degrees in radians
        (4.0, 0.7854, "Correção", "v_dir"),     # Longe, alvo à direita, correção de direção - 45 degrees in radians
        (0.2, 2, "Chegada", "parar"),         # Muito perto do alvo, quase alinhado
        (0.3, 30, "Parada Emerg.", "parar"),  # Obstáculo muito próximo, risco de colisão
        (2.0, -20, "Ajuste Esq.", "v_esq"),   # Alvo à esquerda, sem obstáculos próximos
        (1.0, 10, "Aproximação", "v_dir")     # Distância média, pequeno ajuste à direita
    ]
    print("\nResultados dos Testes de Comportamento do Robô:")
    for dist, angle, label, expected in scenarios:
        action, success = bayesian(dist, angle)
        status = "OK" if action in expected.split('/') else "Inesperado"
        print(f"Cenário ({label:<12}): Ação = {action:<8} | Sucesso = {success:.2%} | Esperado: {expected:<10} | Status: {status}")
