from collections import defaultdict
import json
import pickle
import networkx as nx
from algorithms.greedy import greedy
from algorithms.greedy_fb import greedy_fb
from algorithms.greedy_p import greedyP
from algorithms.rec_greedy import rec_greedy
from sample import Distribuicao, Grafo
import matplotlib.pyplot as plt
from pprint import pprint


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
    if (0, 5) in grafo.arestas or (5, 0) in grafo.arestas:
        return 0
    return 1


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


result_file = open("results/resultado_diamante.json", "rb")
# result_file = open("results/resultado_diamante.json", "wb")
resultados_json = json.loads(pickle.load(result_file))
for alg, resultados in resultados_json.items():
    for num_amostra, sucesso in resultados.items():
        resultados_json[alg][num_amostra] = sucesso / 100

print(resultados_json)
# resultado = get_resultado_diamante()
# pickle.dump(json.dumps(resultado), result_file)


# for n, s in resultados.items():
#     print(f"{n} amostras: { s / 100}")


# dist_d = Distribuicao(tipo="grid", num_amostras=50)
# vizinhanca = rec_greedy(dist_d, 0.13)
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
