from utils_entropy import formata, empirical_cond_entropy
import numpy as np
from sample import Distribuicao

def get_min_j(
    i: int, variaveis: list, dist: Distribuicao, vizinhanca_i: list, non_d: float
):
    melhor_candidato = None
    menor_entropia = np.inf

    for j in variaveis:
        if j != i and j not in vizinhanca_i:
            if len(vizinhanca_i) >= 1:
                lista_input = formata(
                    [dist.amostras[:, v] for v in vizinhanca_i] + [dist.amostras[:, j]]
                )
            else:
                lista_input = [dist.amostras[:, v] for v in vizinhanca_i] + [
                    dist.amostras[:, j]
                ]

            entropia_j = empirical_cond_entropy(
                dist.amostras[:, i],
                lista_input,
            )

            if entropia_j < menor_entropia:
                melhor_candidato = j
                menor_entropia = entropia_j

    return melhor_candidato, menor_entropia


def greedy(dist: Distribuicao, non_d: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {}

    for i in variaveis:
        vizinhanca[i] = set()

        while True:
            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            melhor_candidato, menor_entropia = get_min_j(
                i, variaveis, dist, vizinhanca[i], non_d
            )

            if menor_entropia < entropia_atual - non_d / 2:
                print(
                    f"({i}, {melhor_candidato}): {entropia_atual} - {menor_entropia} = {entropia_atual - menor_entropia}"
                )
                vizinhanca[i].add(melhor_candidato)
            else:
                break

    return vizinhanca
