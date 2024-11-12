import numpy as np
import networkx as nx
from greedy import greedy_algorithm_meu
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

dist_d = Distribuicao(tipo="grid", num_amostras=100)
# mean = np.mean(dist_d.amostras, axis=1)

# plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()

vizinhanca = greedy_algorithm_meu(dist_d, 0.04)
grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)

# dist_d = Distribuicao(tipo="grid", num_amostras=1)
# grafo = dist_d._grafos[0]
# plotar_grafo(grafo.vertices, grafo.arestas)
print(grafo.__dict__)
