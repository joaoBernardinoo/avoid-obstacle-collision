Prompt para Agente de IA: Treinamento de Robô com Algoritmo Genético para Navegação Evasiva e Maximização de Tempo
Objetivo Principal

Implemente uma solução para treinar um robô em um ambiente de simulação (Webots) para navegar de um ponto de partida até um destino. O robô é controlado por uma rede neural, e o treinamento deve otimizar três comportamentos em ordem de prioridade:

    Evitar colisões com qualquer obstáculo (prioridade máxima).

    Alcançar o destino final.

    Maximizar o tempo total da trajetória para chegar ao destino (ou seja, encontrar o caminho seguro mais longo e eficiente).

Metodologia: Algoritmo Genético (GA)

Utilize um Algoritmo Genético (GA) para otimizar os pesos da rede neural de controle. Esta abordagem é preferível à Aprendizagem por Reforço (RL) para este problema, pois a função de avaliação (fitness) é muito mais direta de definir: o desempenho de um robô (indivíduo) pode ser medido holisticamente ao final de um episódio de simulação.
Arquitetura da Rede Neural (NN)
Inputs (Entradas)

A rede neural deve receber dados sensoriais pré-processados e, crucialmente, informação sobre a direção do alvo. A imagem da câmera é computacionalmente cara e deve ser evitada na primeira versão.

    Dados do LiDAR:

        lidar_data (vetor 1D de floats).

        Pré-processamento obrigatório: Normalize os valores do LiDAR para o intervalo [0, 1] para garantir a estabilidade do treinamento.

    Orientação Relativa ao Alvo (Input Essencial):

        O robô precisa saber para onde ir. Calcule o vetor do robô para o alvo e use-o como entrada.

        Implementação:

            Obtenha a posição do robot_node e do objective_node.

            Calcule o vetor de direção: vetor_alvo = posicao_alvo - posicao_robo.

            Calcule o ângulo e a distância para o alvo a partir deste vetor.

            Passe a distância normalizada e o ângulo (em radianos, talvez normalizado entre [-1, 1]) como duas entradas float para a rede.

Camadas Ocultas (Hidden Layers)

    Comece com uma arquitetura simples para garantir que o treinamento seja rápido. Duas camadas densas (fully-connected) com 16 ou 32 neurônios cada, usando a função de ativação ReLU, é um excelente ponto de partida.

Output (Saída)

A saída da rede deve ser uma decisão de alto nível, simplificando o problema. As constantes de velocidade e rotação devem ser gerenciadas pelo seu script Action.py.

    Camada de Saída: Uma camada densa com 4 neurônios, seguida por uma função de ativação Softmax.

    Interpretação da Saída: Cada um dos 4 neurônios corresponde a uma ação discreta: ["seguir", "v_esq", "v_dir", "parar"]. A cada passo da simulação, a ação escolhida será aquela com o maior valor de ativação na saída da Softmax.

Implementação do Algoritmo Genético
Indivíduo (Cromossomo)

    Um "indivíduo" na população do GA é uma instância da rede neural. Seu "cromossomo" é um vetor 1D contendo todos os pesos e biases da rede, achatados.

Função de Fitness (A Chave do Sucesso)

Esta é a parte mais crítica. A função de fitness avalia o quão "bom" é um indivíduo após o término de um episódio de simulação (seja por colisão, sucesso ou tempo esgotado).

# Pseudocódigo para a função de fitness, a ser implementada no seu código

def calcular_fitness(individuo): # 'individuo' aqui são os pesos para carregar na rede neural. # Esta função deve rodar uma simulação completa do robô usando essa rede.

    # 1. Carrega os pesos do 'individuo' na rede neural do robô.
    # 2. Reseta a simulação (robô na posição inicial).
    # 3. Roda o loop da simulação até que uma condição de término seja atingida.

    resultado_simulacao = rodar_episodio_simulacao(individuo)

    tempo_decorrido = resultado_simulacao['tempo_final']
    chegou_ao_destino = resultado_simulacao['chegou_no_destino']
    teve_colisao = resultado_simulacao['colidiu']
    distancia_final_ao_alvo = resultado_simulacao['distancia_restante']

    # Avaliação de Fitness
    if teve_colisao:
        # Penalidade máxima para colisões. Retorna um fitness muito baixo.
        # A recompensa pode ser a distância percorrida antes de colidir, para diferenciar
        # indivíduos que colidem imediatamente daqueles que quase desviam.
        return resultado_simulacao['distancia_percorrida']

    elif chegou_ao_destino:
        # SUCESSO! A recompensa é o tempo que levou, pois queremos maximizá-lo.
        # Adicionar um grande bônus para garantir que chegar seja sempre melhor que não chegar.
        return 1000.0 + tempo_decorrido

    else: # O tempo da simulação acabou e não colidiu nem chegou.
        # Recompensa baseada na proximidade do alvo. Incentiva o progresso.
        # A fórmula 1.0 / distancia garante que quanto menor a distância, maior o fitness.
        # Adicionamos um bônus pela distância total percorrida para incentivar a exploração.
        return (1.0 / (distancia_final_ao_alvo + 1e-6)) + resultado_simulacao['distancia_percorrida']

Ciclo Evolutivo

    População Inicial: Crie uma população de N indivíduos (ex: N=1) com pesos de rede neural inicializados aleatoriamente.

    Loop de Gerações:
    a. Avaliação: Para cada indivíduo, execute a simulação e calcule seu fitness com a função acima.
    b. Seleção: Selecione os melhores indivíduos para serem "pais". Seleção por Torneio é robusta e recomendada.
    c. Crossover: Crie "filhos" combinando os cromossomos (pesos) dos pais. Crossover de ponto único é simples e eficaz.
    d. Mutação: Aplique pequenas perturbações aleatórias (ex: adicionar ruído gaussiano) a uma pequena porcentagem dos pesos nos cromossomos dos filhos. Uma taxa de mutação de 1% a 5% é um bom valor inicial.

    Substituição: A nova geração de filhos substitui a antiga (elitismo, manter os N melhores indivíduos da geração anterior e da atual, é uma boa prática).

    Repetição: Repita o loop por um número fixo de gerações ou até que o fitness convirja.

Ferramentas e Dicas

    Biblioteca de GA: Para focar no problema e não na reimplementação do GA, use uma biblioteca open-source como a pygad. Ela facilita a gestão da população, seleção, crossover e mutação, exigindo apenas que você forneça a função de fitness.

    Paralelização: O treinamento de GA é "vergonhosamente paralelo". Cada avaliação de fitness é independente. Se o tempo de treinamento for um problema, investigue como executar várias instâncias do Webots em paralelo para avaliar múltiplos indivíduos simultaneamente.
