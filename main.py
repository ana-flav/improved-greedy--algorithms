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
        1000: 0.04,
        2000: 0.005,
        3000: 0.005,
        4000: 0.004,
        5000: 0.002,
    }

    for num_amostras in range(1000, 6000, 1000):
        print(num_amostras)
        for i in range(20):
            print(i)
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
            for i in range(4):
                dist_d = Distribuicao(tipo="diamante", num_amostras=num_amostras)
                relacao_algoritmos = {
                    0: (greedy, {"dist": dist_d, "non_d": 0.008}),
                    1: (rec_greedy, {"dist": dist_d, "epsilon": 0.008}),
                    2: (greedy_fb, {"dist": dist_d, "non_d": 0.008, "alpha": 0.9}),
                    3: (greedyP, {"dist": dist_d, "epsilon": 0.008}),
                }
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
        5000: 0.002,
    }

    # Dicionário para armazenar os tempos
    tempos_execucao = {
        "greedy": [],
        "rec_greedy": [],
        "greedy_fb": [],
        "greedyP": [],
    }

    for num_amostras in range(1000, 6000, 1000):
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
        
                if grado.arestas:
                    break

            resultados[alg.__name__][int(num_amostras)] += tempo_exec.total_seconds()
            tempos_execucao[alg.__name__].append(tempo_exec.total_seconds())

    return resultados


def write_resultados_diamante():
    result_file = open("results/resultado_diamante.json", "wb")

    resultado = get_resultado_diamante()
    pickle.dump(json.dumps(resultado), result_file)


def read_resultados_diamante():
    resultado = get_resultado_diamante()
    output = {
        "greedy": defaultdict(int),
        "rec_greedy": defaultdict(int),
        "greedy_fb": defaultdict(int),
        "greedyP": defaultdict(int),
    }
    for alg, resultados in resultado.items():
        for num_amostra, sucesso in resultados.items():
            output[alg][int(num_amostra)] = sucesso / 100

    return output

        
def plotar_tempos_execucao():
    tempos = tempo_execucao()
    df_tempos = pd.DataFrame(tempos)

    plt.figure(figsize=(10, 6))

    # breakpoint()
    for alg in df_tempos:

        plt.plot(df_tempos.index, df_tempos[alg], label=alg, marker="o")

    plt.title("Tempo de Execução dos Algoritmos por Número de Amostras")
    plt.xlabel("Número de Amostras")
    plt.ylabel("Tempo de Execução (segundos)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plotar_taxa_sucesso(res):
    # res = read_resultados_diamante()
    df_sucesso = pd.DataFrame(res)

    plt.figure(figsize=(10, 6))
    for alg in df_sucesso:
        plt.plot(df_sucesso.index, df_sucesso[alg], label=alg, marker="o")

    plt.title("Tempo de Execução dos Algoritmos por Número de Amostras")
    plt.xlabel("Número de Amostras")
    plt.ylabel("Tempo de Execução (segundos)")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_algoritmo(dist, epsilon):
    return {
        0: (greedy, {"dist": dist, "non_d": epsilon}),
        1: (rec_greedy, {"dist": dist, "epsilon": epsilon}),
        2: (greedy_fb, {"dist": dist, "non_d": epsilon, "alpha": 0.9}),
        3: (greedyP, {"dist": dist, "epsilon": epsilon}),
    }

tipo = "grid"
while True:
    epsilon = 0.007 if tipo == "grid" else 0.06
    dist = Distribuicao(tipo=tipo, num_amostras=1000)
    # dist = Distribuicao(tipo="diamante", num_amostras=100)
    
    # vizinhanca = greedy_fb(dist, 0.06, 0.9)
    # vizinhanca = rec_greedy(dist, 0.06)
    vizinhanca = greedyP(dist, 0.007)
    
    grafo = Grafo.get_instance_from_vizinhanca(vizinhanca)
    if grafo.arestas:
        break
    
plotar_grafo(grafo.vertices, grafo.arestas)
