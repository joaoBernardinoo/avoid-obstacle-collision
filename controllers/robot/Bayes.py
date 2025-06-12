import numpy as np


def bayesInfer(dist: float, angle: float) -> tuple[str, float]:
    """
    Realiza inferência na Rede Bayesiana para determinar Ação e Sucesso.
    Args:
        dist (float): Distância estimada ao obstáculo.
        angle (float): Ângulo estimado para o alvo em graus.
    Returns:
        tuple[str, float]: Ação mais provável e probabilidade de sucesso.
    """
    # --- Configuração de estados e parâmetros ---
    s = {
        'O': {'sim': 0, 'nao': 1}, # Obstacle
        'V': {'sim': 0, 'nao': 1}, # Visible
        'D': {'esq': 0, 'fr': 1, 'dir': 2}, # Direction
        'A': {'seguir': 0, 'v_esq': 1, 'v_dir': 2, 'parar': 3} # Action
    }
    action_map = {v: k for k, v in s['A'].items()}

    DIST_NEAR = 0.8
    DIST_STOP = 0.35
    ANGLE_FRONT = 15.0

    # --- 1. Calcular probabilidades das evidências ---
    p_obs_sim = 1 / (1 + np.exp((dist - DIST_NEAR) * 5))
    P_O = np.array([p_obs_sim, 1 - p_obs_sim])

    p_vis_sim = 1.0 if abs(angle) < 45.0 else 0.1
    P_V = np.array([p_vis_sim, 1 - p_vis_sim])

    if angle < -ANGLE_FRONT: P_D = np.array([0.8, 0.1, 0.1])
    elif angle > ANGLE_FRONT: P_D = np.array([0.1, 0.1, 0.8])
    else: P_D = np.array([0.1, 0.8, 0.1])

    # --- 2. Definir Tabelas de Probabilidade Condicional (CPTs) ---
    # CPT para P(Ação | Obstáculo, Visível, Direção)
    cpt_A = np.zeros((2, 2, 3, 4)) # Shape: O, V, D, A
    cpt_A[s['O']['sim'], :, s['D']['esq']] = [[0.1, 0.8, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]]
    cpt_A[s['O']['sim'], :, s['D']['fr']]  = [[0.1, 0.1, 0.1, 0.7], [0.1, 0.4, 0.4, 0.1]] # Parada especial se alvo visível
    cpt_A[s['O']['sim'], :, s['D']['dir']] = [[0.1, 0.1, 0.8, 0.0], [0.1, 0.1, 0.8, 0.0]]
    cpt_A[s['O']['nao'], s['V']['sim'], s['D']['esq']] = [0.1, 0.8, 0.1, 0.0]
    cpt_A[s['O']['nao'], s['V']['sim'], s['D']['fr']]  = [0.9, 0.05, 0.05, 0.0]
    cpt_A[s['O']['nao'], s['V']['sim'], s['D']['dir']] = [0.1, 0.1, 0.8, 0.0]
    cpt_A[s['O']['nao'], s['V']['nao'], :] = [0.3, 0.3, 0.3, 0.1] # Explorar se alvo não visível

    # CPT para P(Sucesso | Ação, Visível)
    cpt_S = np.zeros((4, 2, 2)) # Shape: A, V, S (sim/nao)
    cpt_S[s['A']['seguir']] = [[0.9, 0.1], [0.4, 0.6]]
    cpt_S[s['A']['v_esq']]  = [[0.7, 0.3], [0.6, 0.4]]
    cpt_S[s['A']['v_dir']]  = [[0.7, 0.3], [0.6, 0.4]]
    cpt_S[s['A']['parar']]   = [[0.99, 0.01],[0.1, 0.9]]

    # --- 3. Inferência Bayesiana ---
    # Calcula P(A) marginalizando O, V, D
    P_A = np.einsum('ovda,o,v,d->a', cpt_A, P_O, P_V, P_D)
    P_A /= np.sum(P_A)
    
    action_idx = np.argmax(P_A)
    action_str = action_map[action_idx]

    # Sobrescreve para parada com base na distância bruta para maior robustez
    if dist < DIST_STOP and abs(angle) < ANGLE_FRONT:
        action_str = "parar"
        action_idx = s['A']['parar']

    # Calcula P(S) marginalizando A, V
    P_S = np.einsum('avs,a,v->s', cpt_S, P_A, P_V)
    p_success = P_S[0] # P(Sucesso=sim)

    return action_str, p_success

# --- Demonstração ---
if __name__ == '__main__':
    acao1, suc1 = bayesInfer(dist=5.0, angle=0)
    print(f"Cenário 1 (Livre):    Ação = {acao1:<8} | Sucesso = {suc1:.2%}")

    acao2, suc2 = bayesInfer(dist=0.6, angle=5)
    print(f"Cenário 2 (Perto):    Ação = {acao2:<8} | Sucesso = {suc2:.2%}")

    acao3, suc3 = bayesInfer(dist=4.0, angle=45)
    print(f"Cenário 3 (Corrigir): Ação = {acao3:<8} | Sucesso = {suc3:.2%}")

    acao4, suc4 = bayesInfer(dist=0.2, angle=2)
    print(f"Cenário 4 (Chegada):  Ação = {acao4:<8} | Sucesso = {suc4:.2%}")