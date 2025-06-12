import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definir a estrutura da Rede Bayesiana (nós e arestas/dependências)
# A estrutura reflete as dependências causais que definimos antes.
model = DiscreteBayesianNetwork([
    ('Obstacle', 'Action'),
    ('Visible', 'Action'),
    ('Direction', 'Action'),
    ('Action', 'Success') # O sucesso em chegar ao destino depende da sequência de ações
])

# CPT para a Ação. Depende de Obstacle, Visible e Direction.
# 2. Definir as Tabelas de Probabilidade Condicional (CPTs)

# --- CPTs para os Nós Raiz (A CORREÇÃO ESTÁ AQUI) ---
# Usamos um valor padrão uniforme. Eles serão sobrescritos pela evidência.
cpt_O = TabularCPD(variable='Obstacle', variable_card=2, values=[[0.5], [0.5]])
cpt_V = TabularCPD(variable='Visible', variable_card=2, values=[[0.5], [0.5]])
cpt_D = TabularCPD(variable='Direction', variable_card=3, values=[[0.33], [0.34], [0.33]])

# A ordem das evidências em 'evidence' é crucial e deve corresponder ao shape do array 'values'.
cpt_A = TabularCPD(
    variable='Action',
    variable_card=4, # 4 estados: seguir, v_esq, v_dir, parar
    values=[
        # Tabela de probabilidade, cada coluna é uma combinação das evidências
        # O=s,V=s,D=e | O=s,V=s,D=f | O=s,V=s,D=d | O=s,V=n,D=e | ...
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.90, 0.10, 0.10, 0.30, 0.30, 0.30], # seguir
        [0.80, 0.10, 0.10, 0.80, 0.40, 0.10, 0.05, 0.80, 0.10, 0.30, 0.30, 0.30], # v_esq
        [0.10, 0.10, 0.80, 0.10, 0.40, 0.80, 0.05, 0.10, 0.80, 0.30, 0.30, 0.30], # v_dir
        [0.00, 0.70, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.10, 0.10, 0.10]  # parar
    ],
    evidence=['Obstacle', 'Visible', 'Direction'],
    evidence_card=[2, 2, 3] # O(2), V(2), D(3)
)

# CPT para o Sucesso. Por sua definição, ele depende da ação tomada.
# A probabilidade de sucesso é a chance de eventualmente chegar no destino.
cpt_S = TabularCPD(
    variable='Success',
    variable_card=2, # 2 estados: sim, nao
    values=[
        # Prob(Success=sim | Action)
        # seguir, v_esq, v_dir, parar
        [0.90,   0.60,   0.60,  0.99], # Probabilidade de sucesso 'sim'
        [0.10,   0.40,   0.40,  0.01]  # Probabilidade de sucesso 'nao'
    ],
    evidence=['Action'],
    evidence_card=[4]
)

# Adicionar as CPTs ao modelo
model.add_cpds(cpt_A, cpt_S,cpt_D,cpt_O,cpt_V)

# Verificar se o modelo e as CPTs são consistentes
if not model.check_model():
    raise ValueError("O modelo Bayesiano não é válido.")

# Criar um objeto de inferência
inference = VariableElimination(model)

# --- Função que usa o modelo pgmpy ---
def bayesInfer_pgmpy(dist: float, angle: float) -> tuple[str, float]:
    """Usa o modelo pgmpy para inferir a ação e a probabilidade de sucesso."""
    # Mapeamento de nomes para índices, para facilitar a leitura
    action_map = {0: 'seguir', 1: 'v_esq', 2: 'v_dir', 3: 'parar'}
    
    # Parâmetros de configuração
    DIST_NEAR = 0.8
    DIST_STOP = 0.35
    ANGLE_FRONT = 15.0

    # Calcular as probabilidades de evidência (soft evidence) com base nas entradas
    p_obs_sim = 1 / (1 + np.exp((dist - DIST_NEAR) * 5))
    p_vis_sim = 1.0 if abs(angle) < 45.0 else 0.1
    if angle < -ANGLE_FRONT: p_dir = [0.8, 0.1, 0.1]
    elif angle > ANGLE_FRONT: p_dir = [0.1, 0.1, 0.8]
    else: p_dir = [0.1, 0.8, 0.1]

    # As evidências são as probabilidades para os nós raiz
    evidence_dict = {
        'Obstacle': [p_obs_sim, 1 - p_obs_sim],
        'Visible': [p_vis_sim, 1 - p_vis_sim],
        'Direction': p_dir
    }
    
    # Inferir a ação mais provável (MAP - Maximum a Posteriori)
    map_result = inference.map_query(variables=['Action'], evidence=evidence_dict)
    action_str = map_result['Action']

    # Sobrescrever para parada com base na distância bruta
    if dist < DIST_STOP and abs(angle) < ANGLE_FRONT:
        action_str = "parar"
        
    # Inferir a probabilidade de sucesso DADA a ação que será tomada
    # Para isso, adicionamos a ação inferida como evidência (hard evidence)
    final_evidence = evidence_dict.copy()
    final_evidence['Action'] = list(action_map.keys())[list(action_map.values()).index(action_str)]

    prob_success_dist = inference.query(variables=['Success'], evidence=final_evidence)
    p_success = prob_success_dist.values[0] # P(Success=sim)
    
    return action_str, p_success

# --- Demonstração ---
if __name__ == '__main__':
    acao1, suc1 = bayesInfer_pgmpy(dist=5.0, angle=0)
    print(f"Cenário 1 (Livre):    Ação = {acao1:<8} | Sucesso = {suc1:.2%}")

    acao2, suc2 = bayesInfer_pgmpy(dist=0.6, angle=5)
    print(f"Cenário 2 (Desvio):   Ação = {acao2:<8} | Sucesso = {suc2:.2%}")

    acao3, suc3 = bayesInfer_pgmpy(dist=4.0, angle=45)
    print(f"Cenário 3 (Correção): Ação = {acao3:<8} | Sucesso = {suc3:.2%}")

    acao4, suc4 = bayesInfer_pgmpy(dist=0.2, angle=2)
    print(f"Cenário 4 (Chegada):  Ação = {acao4:<8} | Sucesso = {suc4:.2%}")