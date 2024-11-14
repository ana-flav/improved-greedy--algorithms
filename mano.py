import numpy as np
from collections import Counter

from mostra import GraphPlotter

def calcular_entropia(probabilidades):
    """
    Calcula a entropia dada uma lista de probabilidades.
    """
    return -np.sum([p * np.log2(p) for p in probabilidades if p > 0])

def calcular_entropia_condicional(dados, variavel_x, variaveis_y):
    """
    Calcula a entropia condicional H(X | Y1, Y2, ..., Yn) usando frequências em um conjunto de dados discreto.
    
    Parâmetros:
    - dados: uma matriz numpy onde cada coluna representa uma variável.
    - variavel_x: índice da variável X (inteiro).
    - variaveis_y: lista de índices das variáveis Y (lista de inteiros).
    
    Retorna:
    - A entropia condicional H(X | Y1, Y2, ..., Yn).
    """
    x = dados[:, variavel_x]
    y = dados[:, variaveis_y]  # agora Y é um conjunto de variáveis
    
    total_amostras = len(x)
    
    # Inicializa a entropia condicional
    entropia_condicional = 0.0
    
    # Calcula a entropia condicional em relação ao conjunto de variáveis Y
    contador_y = Counter(map(tuple, y))  # Conta as combinações dos valores de Y
    for y_valores, freq_y in contador_y.items():
        # Filtra os dados de X quando as variáveis Y possuem uma combinação específica
        x_dado_y = x[np.all(y == np.array(y_valores), axis=1)]
        contador_x_dado_y = Counter(x_dado_y)
        
        # Probabilidades de X dado a combinação específica de Y
        probabilidades_x_dado_y = [freq / freq_y for freq in contador_x_dado_y.values()]
        entropia_x_dado_y = calcular_entropia(probabilidades_x_dado_y)
        
        # Contribuição da combinação de Y para a entropia condicional
        entropia_condicional += (freq_y / total_amostras) * entropia_x_dado_y
    
    return entropia_condicional

def greedy_entropia_condicional_4x4(dados, epsilon):
    n_variaveis = dados.shape[1]
    estrutura_vizinhanca = {i: set() for i in range(n_variaveis)}

    def vizinhos_grade_4x4(i):
        posicoes = []
        if i % 4 != 0:
            posicoes.append(i - 1)
        if (i + 1) % 4 != 0:
            posicoes.append(i + 1)
        if i - 4 >= 0:
            posicoes.append(i - 4)
        if i + 4 < 16:
            posicoes.append(i + 4)
        return posicoes

    for i in range(n_variaveis):
        H_Xi = calcular_entropia([count / len(dados) for count in Counter(dados[:, i]).values()])

        for j in vizinhos_grade_4x4(i):
            # Agora, ao invés de considerar apenas uma variável Xj, considera-se um conjunto de variáveis Y
            variaveis_y = [j]  # Aqui estamos considerando a variável j, mas pode ser um conjunto de variáveis
            H_Xi_dado_Xj = calcular_entropia_condicional(dados, variavel_x=i, variaveis_y=variaveis_y)
            
            delta_H = H_Xi - H_Xi_dado_Xj
            
            if delta_H >= epsilon / 2:
                estrutura_vizinhanca[i].add(j)

    return estrutura_vizinhanca


# Exemplo de uso
# Gerando dados aleatórios para 16 variáveis em uma grade 4x4 (100 amostras)
np.random.seed(0)  # Para reprodutibilidade
dados = np.random.randint(0, 2, size=(100, 16))  # Dados binários (0 ou 1)

# Parâmetro epsilon para o limiar de redução de entropia
epsilon = 0.006

# Executa o algoritmo Greedy com entropia condicional para a grade 4x4
estrutura = greedy_entropia_condicional_4x4(dados, epsilon)

# Exibe a estrutura de vizinhança aprendida
print("Estrutura de Vizinhança Aprendida (Grafo 4x4):")
for i, vizinhos in estrutura.items():
    print(f"Variável {i}: Vizinhos -> {list(vizinhos)}")

# Após executar o algoritmo greedy
plotter = GraphPlotter(4, 4)
plotter.plot_graph_structure(estrutura, dados)

