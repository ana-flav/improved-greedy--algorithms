import numpy as np
import networkx as nx
from greedy import greedy_algorithm_meu
from teste import cross_validate_non_d
from utils import Distribuicao, Grafo
import matplotlib.pyplot as plt


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

dist_d = Distribuicao(tipo="grid", num_amostras=1000)
# mean = np.mean(dist_d.amostras, axis=1)

# plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()

# for epsilon in [0.04, 0.02, 0.01, 0.014, 0.009, 0.008, 0.006, 0.003]:
vizinhanca = greedy_algorithm_meu(dist_d, 0.004)
grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)

print(grafo.__dict__)

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

