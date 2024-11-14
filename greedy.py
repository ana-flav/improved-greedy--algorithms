from collections import Counter, defaultdict
import math
import numpy as np
from utils import Distribuicao

def formata(listao):
    pares = []
    for linha in range(len(listao[0])):
        pares.append([l[linha] for l in listao])
    return pares


def empirical_cond_entropy(X, Y=[]):
    if len(Y) == 0:
        count = Counter(X)
        n = len(X)

        prob = [count[val] / n for val in count]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    if len(Y) > 1:
        Y = formata(Y)

    X = np.array(X)

    if isinstance(Y, list) and isinstance(Y[0], (list, tuple)):
        Y = np.array(Y)
    else:
        Y = np.array(Y).reshape(-1, 1)

    y_to_x = defaultdict(list)
    for i in range(len(X)):
        y_val = tuple(Y[i]) if Y.shape[1] > 1 else Y[i][0]
        y_to_x[y_val].append(X[i])

    n_samples = len(X)
    conditional_entropy = 0

    for y_val, x_vals in y_to_x.items():
        p_y = len(x_vals) / n_samples

        x_counts = defaultdict(int)
        for x in x_vals:
            x_counts[x] += 1

        entropy_xy = 0
        for x_count in x_counts.values():
            p_x_given_y = x_count / len(x_vals)
            entropy_xy -= p_x_given_y * np.log2(p_x_given_y)

        conditional_entropy += p_y * entropy_xy

    return conditional_entropy


def get_min_j(
    i: int, variaveis: list, dist: Distribuicao, vizinhanca_i: list, non_d: float
):
    melhor_candidato = None
    menor_entropia = np.inf

    for j in variaveis:
        if j != i and j not in vizinhanca_i:
            entropia_j = empirical_cond_entropy(
                dist.amostras[:, i],
                [dist.amostras[:, v] for v in vizinhanca_i] + [dist.amostras[:, j]],
            )

            if entropia_j < menor_entropia:
                melhor_candidato = j
                menor_entropia = entropia_j

    return melhor_candidato, menor_entropia


def greedy(dist: Distribuicao, non_d: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {i: set() for i in variaveis}

    for i in variaveis:
        while True:
            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            melhor_candidato, menor_entropia = get_min_j(
                i, variaveis, dist, vizinhanca[i], non_d
            )

            if menor_entropia < entropia_atual - non_d / 2:
                vizinhanca[i].add(melhor_candidato)
                vizinhanca[melhor_candidato].add(i)
            else:
                break

    return vizinhanca
