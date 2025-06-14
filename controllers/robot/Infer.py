import numpy as np
import logging
import cProfile
import pstats
from line_profiler import LineProfiler
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os
import time
import pandas as pd

logging.getLogger("pgmpy").setLevel(logging.ERROR)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bayes_model.bif')

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = DiscreteBayesianNetwork.load(MODEL_PATH)
    print("[INFO] Modelo Bayesiano carregado de bayes_model")
else:
    # 1. Definir a estrutura da Rede Bayesiana (nós e arestas/dependências)
    # A estrutura reflete as dependências causais que definimos antes.
    model = DiscreteBayesianNetwork([
        ('Obstacle', 'Action'),
        ('Visible', 'Action'),
        ('Direction', 'Action'),
        # O sucesso em chegar ao destino depende da sequência de ações
        ('Action', 'Success')
    ])

    # CPT para a Ação. Depende de Obstacle, Visible e Direction.
    # 2. Definir as Tabelas de Probabilidade Condicional (CPTs)
    # --- CPTs para os Nós Raiz (A CORREÇÃO ESTÁ AQUI) ---
    # Usamos um valor padrão uniforme. Eles serão sobrescritos pela evidência.
    cpt_O = TabularCPD(
        variable='Obstacle',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Obstacle': ['sim', 'nao']}
    )

    cpt_V = TabularCPD(
        variable='Visible',
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={'Visible': ['sim', 'nao']}
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
            [0.20, 0.26, 0.10, 0.10, 0.10, 0.10, 0.90,
                0.10, 0.10, 0.30, 0.30, 0.30],  # seguir
            [0.70, 0.07, 0.10, 0.80, 0.40, 0.10, 0.05,
                0.80, 0.10, 0.30, 0.30, 0.30],  # v_esq
            [0.10, 0.07, 0.80, 0.10, 0.40, 0.80, 0.05,
                0.10, 0.80, 0.30, 0.30, 0.30],  # v_dir
            [0.00, 0.60, 0.00, 0.00, 0.10, 0.00, 0.00,
                0.00, 0.00, 0.10, 0.10, 0.10]  # parar
        ],
        evidence=['Obstacle', 'Visible', 'Direction'],
        evidence_card=[2, 2, 3],  # O(2), V(2), D(3)
        state_names={
            'Action': ['seguir', 'v_esq', 'v_dir', 'parar'],
            'Obstacle': ['sim', 'nao'],
            'Visible': ['sim', 'nao'],
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


# --- Função que usa o modelo pgmpy ---


def bayesian(dist: float, angle: float) -> tuple[str, float]:
    """Usa o modelo pgmpy para inferir a ação e a probabilidade de sucesso com soft evidence."""
    action_map = {0: 'seguir', 1: 'v_esq', 2: 'v_dir', 3: 'parar'}

    DIST_NEAR = 0.8
    DIST_STOP = 0.35
    ANGLE_FRONT = 15.0

    # Soft evidence (probabilidades)
    p_obs_sim = 1 / (1 + np.exp((dist - DIST_NEAR) * 5))
    p_obs = [p_obs_sim, 1 - p_obs_sim]
    p_vis_sim = 1.0 if abs(angle) < 45.0 else 0.1
    p_vis = [p_vis_sim, 1 - p_vis_sim]
    if angle < -ANGLE_FRONT:
        p_dir = [0.8, 0.1, 0.1]
    elif angle > ANGLE_FRONT:
        p_dir = [0.1, 0.1, 0.8]
    else:
        p_dir = [0.1, 0.8, 0.1]

    # Virtual evidence: lista de fatores (um para cada variável)
    virtual_evidence = [
        TabularCPD('Obstacle', 2, [[p_obs[0]], [p_obs[1]]], 
                  state_names={'Obstacle': ['sim', 'nao']}),
        TabularCPD('Visible', 2, [[p_vis[0]], [p_vis[1]]],
                  state_names={'Visible': ['sim', 'nao']}),
        TabularCPD('Direction', 3, [[p_dir[0]], [p_dir[1]], [p_dir[2]]],
                  state_names={'Direction': ['esquerda', 'frente', 'direita']})
    ]

    # Inferir a distribuição de Action com soft evidence
    action_dist = inference.query(
        variables=['Action'], virtual_evidence=virtual_evidence) # type: ignore
    action_idx = int(np.argmax(action_dist.values)) # type: ignore
    action_str = action_map[action_idx]

    if dist < DIST_STOP and abs(angle) < ANGLE_FRONT:
        action_str = "parar"
        action_idx = 3

    # Probabilidade de sucesso para a ação escolhida
    prob_success_dist = inference.query(
        variables=['Success'],
        evidence={'Action': action_str}  # Use state name instead of index
    ) # type: ignore
    p_success = prob_success_dist.values[0] # type: ignore

    return action_str, p_success # type: ignore


def profile_bayesian():
    """Function to profile the bayesian inference with different scenarios"""
    scenarios = [
        (5.0, 0),    # Cenário 1 (Livre)
        (0.6, 5),    # Cenário 2 (Desvio)
        (4.0, 45),   # Cenário 3 (Correção)
        (0.2, 2),    # Cenário 4 (Chegada)
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
    acao1, suc1 = bayesian(dist=5.0, angle=0)
    acao2, suc2 = bayesian(dist=0.6, angle=5)
    acao3, suc3 = bayesian(dist=4.0, angle=45)
    acao4, suc4 = bayesian(dist=0.2, angle=2)

    print(f"\nCenário 1 (Livre):    Ação = {acao1:<8} | Sucesso = {suc1:.2%}")
    print(f"Cenário 2 (Desvio):   Ação = {acao2:<8} | Sucesso = {suc2:.2%}")
    print(f"Cenário 3 (Correção): Ação = {acao3:<8} | Sucesso = {suc3:.2%}")
    print(f"Cenário 4 (Chegada):  Ação = {acao4:<8} | Sucesso = {suc4:.2%}")
    
