from utils_entropy import empirical_cond_entropy
import numpy as np
from sample import Distribuicao


def get_min_j(i: int, variaveis: list, dist: Distribuicao, vizinhanca_i: list):
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
                i, variaveis, dist, vizinhanca[i]
            )

            if menor_entropia < entropia_atual - non_d / 2:
                vizinhanca[i].add(melhor_candidato)
            else:
                break

    return vizinhanca
