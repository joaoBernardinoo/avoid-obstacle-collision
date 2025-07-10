from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
import os
import numpy as np


def load_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bayes_model')

    if os.path.exists(MODEL_PATH):
        # model = DiscreteBayesianNetwork.load(MODEL_PATH + ".bif")
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
        # values = [
        #     # seguir, v_esq, v_dir, parar
        #     [0.15, 0.45, 0.00, 0.40],  # ( sim, sim, esquerda )
        #     [0.40, 0.00, 0.00, 0.60],  # ( sim, sim, frente )
        #     [0.15, 0.00, 0.45, 0.40],  # ( sim, sim, direita )
        #     [0.10, 0.80, 0.10, 0.00],  # ( sim, nao, esquerda )
        #     [0.10, 0.45, 0.45, 0.00],  # ( sim, nao, frente )
        #     [0.10, 0.10, 0.80, 0.00],  # ( sim, nao, direita )
        #     [0.80, 0.00, 0.00, 0.20],  # ( nao, sim, esquerda )
        #     [0.90, 0.00, 0.00, 0.10],  # ( nao, sim, frente )
        #     [0.80, 0.00, 0.00, 0.20],  # ( nao, sim, direita )
        #     [0.10, 0.50, 0.30, 0.10],  # ( nao, nao, esquerda )
        #     [0.40, 0.25, 0.25, 0.10],  # ( nao, nao, frente )
        #     [0.10, 0.30, 0.50, 0.10]   # ( nao, nao, direita )
        # ]

        values = [
            # seguir, v_esq, v_dir, parar
            [0.20, 0.50, 0.00, 0.30],  # ( sim, sim, esquerda )
            [0.30, 0.00, 0.00, 0.70],  # ( sim, sim, frente )
            [0.20, 0.00, 0.50, 0.30],  # ( sim, sim, direita )
            [0.10, 0.80, 0.10, 0.00],  # ( sim, nao, esquerda )
            [0.10, 0.45, 0.45, 0.00],  # ( sim, nao, frente )
            [0.10, 0.10, 0.80, 0.00],  # ( sim, nao, direita )
            [0.90, 0.10, 0.00, 0.00],  # ( nao, sim, esquerda )
            [1.00, 0.00, 0.00, 0.00],  # ( nao, sim, frente )
            [0.90, 0.00, 0.10, 0.00],  # ( nao, sim, direita )
            [0.10, 0.50, 0.30, 0.10],  # ( nao, nao, esquerda )
            [0.40, 0.25, 0.25, 0.10],  # ( nao, nao, frente )
            [0.10, 0.30, 0.50, 0.10]   # ( nao, nao, direita )
        ]

        # A ordem das evidências em 'evidence' é crucial e deve corresponder ao shape do array 'values'.
        cpt_A = TabularCPD(
            variable='Action',
            variable_card=4,  # 4 estados: seguir, v_esq, v_dir, parar
            values=np.array(values).T,
            evidence=['ObstacleDetected', 'TargetVisible', 'Direction'],
            evidence_card=[2, 2, 3],  # O(2), V(2), D(3)
            state_names={
                'Action': ['seguir', 'v_esq', 'v_dir', 'parar'],
                'ObstacleDetected': ['sim', 'nao'],
                'TargetVisible': ['sim', 'nao'],
                'Direction': ['esquerda', 'frente', 'direita']
            }
        )

        cpt_S = TabularCPD(
            variable='Success',
            variable_card=2,  # 2 estados: sim, nao
            values=[
                # Probabilidade de sucesso 'sim'
                [0.90,   0.60,   0.60,  0.99],
                [0.10,   0.40,   0.40,  0.01]  # Probabilidade de sucesso 'nao'
            ],
            evidence=['Action'],
            evidence_card=[4],
            state_names={
                'Success': ['sim', 'nao'],
                'Action': ['seguir', 'v_esq', 'v_dir', 'parar']
            }
        )

        # Adicionar as CPTs ao modelo
        model.add_cpds(cpt_A, cpt_S, cpt_D, cpt_O, cpt_V)
        # Verificar se o modelo e as CPTs são consistentes

        model.save(MODEL_PATH+".bif")
        model.save(MODEL_PATH, filetype='xmlbif')
        if not model.check_model():
            raise ValueError("O modelo Bayesiano não é válido.")
        print("[INFO] Modelo Bayesiano salvo em bayes_model")

    # Criar um objeto de inferência
    inference = VariableElimination(model)
    return inference
