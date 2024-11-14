import numpy as np
from algorithms.greedy import get_min_j
from sample import Distribuicao
from utils_entropy import empirical_cond_entropy


def get_worst_neighbor(i: int, dist: Distribuicao, vizinhanca_i: set):
    pior_vizinho = None
    menor_entropia = np.inf

    for j in vizinhanca_i:
        entropia_sem_j = empirical_cond_entropy(
            dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca_i if v != j]
        )
        if entropia_sem_j < menor_entropia:
            pior_vizinho = j
            menor_entropia = entropia_sem_j

    return pior_vizinho, menor_entropia


def greedy_fb(dist: Distribuicao, non_d: float, alpha: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {i: set() for i in variaveis}

    for i in variaveis:
        while True:
            # Forward
            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            melhor_candidato, menor_entropia = get_min_j(
                i, variaveis, dist, vizinhanca[i]
            )

            if menor_entropia < entropia_atual - non_d / 2:
                vizinhanca[i].add(melhor_candidato)
                adicionado = True
            else:
                adicionado = False

            # Backward
            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            pior_vizinho, menor_entropia = get_worst_neighbor(i, dist, vizinhanca[i])

            if menor_entropia - entropia_atual <= (alpha * non_d) / 2:
                vizinhanca[i].remove(pior_vizinho)
            else:
                if not adicionado:
                    break

    return vizinhanca
