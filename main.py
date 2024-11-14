from collections import defaultdict
import json
import pickle
import networkx as nx
import numpy as np
from algorithms.greedy import greedy
from algorithms.greedy_fb import greedy_fb
from algorithms.greedy_p import greedyP
from algorithms.rec_greedy import rec_greedy
from sample import Distribuicao, Grafo
import matplotlib.pyplot as plt
from pprint import pprint
from datetime import datetime

import pandas as pd

def plotar_grafo(vertices, arestas):
    G = nx.Graph()

    G.add_nodes_from(vertices)
    G.add_edges_from(arestas)

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=15,
        font_color="black",
        edge_color="gray",
    )

    plt.show()


def valida_diamante(grafo):
    if not grafo.arestas:
        return 0
    if (0, 5) in grafo.arestas or (5, 0) in grafo.arestas:
        return 0
    return 1


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


def valida_grade(grafo: Grafo):
    if not grafo.arestas:
        return 0

    for i in range(len(grafo.adjacencia)):
        for vizinho in grafo.adjacencia[i]:
            if vizinho not in vizinhos_grade_4x4(i):
                return 0
    return 1


def get_resultado_grade():
    resultados = {
        "greedy": defaultdict(int),
        "rec_greedy": defaultdict(int),
        "greedy_fb": defaultdict(int),
        "greedyP": defaultdict(int),
    }

    epsilon_amostra = {
        1000: 0.08,
        2000: 0.008,
        3000: 0.005,
        4000: 0.06,
        5000: 0.002,
    }

    for num_amostras in range(1000, 6000, 1000):
        print(num_amostras)
        for _ in range(100):
            dist_d = Distribuicao(tipo="grid", num_amostras=num_amostras)
            epsilon = epsilon_amostra[num_amostras]
            relacao_algoritmos = {
                0: (greedy, {"dist": dist_d, "non_d": epsilon}),
                1: (rec_greedy, {"dist": dist_d, "epsilon": epsilon}),
                2: (greedy_fb, {"dist": dist_d, "non_d": epsilon, "alpha": 0.9}),
                3: (greedyP, {"dist": dist_d, "epsilon": epsilon}),
            }

            for i in range(4):
                alg = relacao_algoritmos[i][0]
                args = relacao_algoritmos[i][1]

                vizinhanca = alg(**args)
                resultados[alg.__name__][int(num_amostras)] += valida_grade(
                    Grafo.get_instance_from_vizinhanca(vizinhanca)
                )

    return resultados


def get_resultado_diamante():
    resultados = {
        "greedy": defaultdict(int),
        "rec_greedy": defaultdict(int),
        "greedy_fb": defaultdict(int),
        "greedyP": defaultdict(int),
    }

    for num_amostras in range(100, 1100, 100):
        print(num_amostras)
        for _ in range(100):
            dist_d = Distribuicao(tipo="diamante", num_amostras=num_amostras)
            relacao_algoritmos = {
                0: (greedy, {"dist": dist_d, "non_d": 0.06}),
                1: (rec_greedy, {"dist": dist_d, "epsilon": 0.06}),
                2: (greedy_fb, {"dist": dist_d, "non_d": 0.06, "alpha": 0.9}),
                3: (greedyP, {"dist": dist_d, "epsilon": 0.06}),
            }

            for i in range(4):
                alg = relacao_algoritmos[i][0]
                args = relacao_algoritmos[i][1]

                vizinhanca = alg(**args)
                resultados[alg.__name__][int(num_amostras)] += valida_diamante(
                    Grafo.get_instance_from_vizinhanca(vizinhanca)
                )

    return resultados

def tempo_execucao():
    resultados = {
        "greedy": defaultdict(int),
        "rec_greedy": defaultdict(int),
        "greedy_fb": defaultdict(int),
        "greedyP": defaultdict(int),
    }

    epsilon_amostra = {
        1000: 0.08,
        2000: 0.008,
        3000: 0.005,
        4000: 0.06,
    }

    # Dicionário para armazenar os tempos
    tempos_execucao = {
        "greedy": [],
        "rec_greedy": [],
        "greedy_fb": [],
        "greedyP": [],
    }

    for num_amostras in range(1000, 5000, 1000):
        dist_d = Distribuicao(tipo="grid", num_amostras=num_amostras)
        epsilon = epsilon_amostra[num_amostras]
        relacao_algoritmos = {
            0: (greedy, {"dist": dist_d, "non_d": epsilon}),
            1: (rec_greedy, {"dist": dist_d, "epsilon": epsilon}),
            2: (greedy_fb, {"dist": dist_d, "non_d": epsilon, "alpha": 0.9}),
            3: (greedyP, {"dist": dist_d, "epsilon": epsilon}),
        }

        for i in range(4):
            alg = relacao_algoritmos[i][0]
            args = relacao_algoritmos[i][1]

            while True:
                inicio_tempo = datetime.now()
                vizinhanca = alg(**args)
                tempo_exec = datetime.now() - inicio_tempo
                grado = Grafo.get_instance_from_vizinhanca(vizinhanca)
                print("morte")
                if grado.arestas:
                    break

            resultados[alg.__name__][int(num_amostras)] += tempo_exec.total_seconds()
            # print(tempo_exec.total_seconds())
            tempos_execucao[alg.__name__].append(tempo_exec.total_seconds())

    return resultados


def write_resultados_diamante():
    result_file = open("results/resultado_diamante.json", "wb")

    resultado = get_resultado_diamante()
    pickle.dump(json.dumps(resultado), result_file)


def read_resultados_diamante():
    result_file = open("results/resultado_diamante.json", "rb")
    resultados_json = json.loads(pickle.load(result_file))
    output = {
        "greedy": defaultdict(int),
        "rec_greedy": defaultdict(int),
        "greedy_fb": defaultdict(int),
        "greedyP": defaultdict(int),
    }
    for alg, resultados in resultados_json.items():
        for num_amostra, sucesso in resultados.items():
            output[alg][int(num_amostra)] = sucesso / 100

    pprint(output)


# read_resultados_diamante()

num = 5000
epsilon = 0.002
sucesso = 0
print(num, epsilon)
for i in range(100):
    print(i)

    tentativas = 1
    while True:
        print("tentando dnv")
        dist_d = Distribuicao(tipo="grid", num_amostras=num)
        vizinhanca = greedy(dist_d, epsilon)
        grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)
        if grafo.arestas:
            sucesso += valida_grade(grafo)
            break
        
def plotar_tempos_execucao():
    tempos = tempo_execucao()
    df_tempos = pd.DataFrame(tempos)

    plt.figure(figsize=(10, 6))

    # breakpoint()
    for alg in df_tempos:

        plt.plot(df_tempos.index, df_tempos[alg], label=alg, marker='o')

    plt.title("Tempo de Execução dos Algoritmos por Número de Amostras")
    plt.xlabel("Número de Amostras")
    plt.ylabel("Tempo de Execução (segundos)")
    plt.legend()
    plt.grid(True)
    plt.show()


plotar_tempos_execucao()



# resultados = {
#     "greedy": {},
#     "rec_greedy": {},
#     "greedy_fb": {},
#     "greedyP": {},
# }


def get_algoritmo(dist, epsilon):
    return {
        0: (greedy, {"dist": dist, "non_d": epsilon}),
        1: (rec_greedy, {"dist": dist, "epsilon": epsilon}),
        2: (greedy_fb, {"dist": dist, "non_d": epsilon, "alpha": 0.9}),
        3: (greedyP, {"dist": dist, "epsilon": epsilon}),
    }


# for a in range(4):
#     for epsilon in np.arange(0, 0.16, 0.02):
#         sucesso = 0
#         for i in range(100):
#             dist_d = Distribuicao(tipo="diamante", num_amostras=1000)
#             algoritmo, args = get_algoritmo(dist_d, epsilon)[a]
#             print(algoritmo.__name__, epsilon, i)

#             vizinhanca = algoritmo(**args)
#             grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)
#             sucesso += valida_diamante(grafo)
#         resultados[algoritmo.__name__][epsilon] = sucesso / 100
#         print(resultados)

# file = open("results/sucesso_epsilon.json", "wb")
# pickle.dump(json.dumps(resultados), file)


# dist_d = Distribuicao(tipo="grid", num_amostras=1000)
# vizinhanca = greedy_fb(dist_d, 0.007, 0.9)
# grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)
# pprint(grafo.__dict__)
# mean = np.mean(dist_d.amostras, axis=1)

# plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()

# dist_d = Distribuicao(tipo="grid", num_amostras=1)
# grafo = dist_d._grafos[0]
# plotar_grafo(grafo.vertices, grafo.arestas)
# print(grafo.__dict__)

# Parâmetros e configuração
# dist_d = Distribuicao(tipo="grid", num_amostras=10000)
# mean = np.mean(dist_d.amostras, axis=1)

# Visualizar a distribuição dos dados
# import matplotlib.pyplot as plt
# plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()

# Lista de valores de epsilon para testar
# epsilon_values = [0.00001, 0.00005, 0.0001, 0.001, 0.005]

# Realizar a validação cruzada para encontrar o melhor epsilon
# best_epsilon, best_score = cross_validate_non_d(dist_d, epsilon_values, k=5)
# print(f"Melhor epsilon: {best_epsilon} com score médio: {best_score}")

# Executar o algoritmo com o melhor epsilon encontrado
# vizinhanca = greedy_algorithm_meu(dist_d, 0.0001)
# mrf = Grafo.get_instance_from_vizinhanca(vizinhanca)
# print("Vizinhança encontrada:", mrf.__dict__)

# Após executar o algoritmo greedy
# plotter = GraphPlotter(4, 4)
# plotter.plot_graph_structure(estrutura, dados)
