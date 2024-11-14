import numpy as np
from greedy import get_min_j
from sample import Distribuicao
from utils_entropy import empirical_cond_entropy

brekar = False

def get_worst_neighbor(i: int, dist: Distribuicao, vizinhanca_i: set):
    pior_vizinho = None
    menor_entropia = np.inf

    for j in vizinhanca_i:
        global brekar
        if brekar:
            breakpoint()
        vizinhanca_sem_j = vizinhanca_i.copy()
        vizinhanca_sem_j.remove(j)

        vizinhos_sem_j = [dist.amostras[:, v] for v in vizinhanca_sem_j]

        entropia_sem_j = empirical_cond_entropy(dist.amostras[:, i], vizinhos_sem_j)
        if entropia_sem_j < menor_entropia:
            pior_vizinho = j
            menor_entropia = entropia_sem_j

    return pior_vizinho, menor_entropia


def greedy_fb(dist: Distribuicao, non_d: float, alpha: float):
    global brekar
    variaveis = list(range(dist.tamanho))
    vizinhanca = {i: set() for i in variaveis}

    for i in variaveis:
        while True:
            # Forward Step
            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            melhor_candidato, menor_entropia = get_min_j(
                i, variaveis, dist, vizinhanca[i]
            )

            if menor_entropia < entropia_atual - non_d / 2:
                vizinhanca[i].add(melhor_candidato)
                vizinhanca[melhor_candidato].add(i)
                print('adicionado')
                adicionado = True
                if len(vizinhanca[i]) > 1:
                    brekar = True
            else:
                adicionado = False

            entropia_atual = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )

            print(f"{i}, {vizinhanca[i]}")
            if brekar:
                breakpoint()

            # Backward Step
            pior_vizinho, menor_entropia = get_worst_neighbor(i, dist, vizinhanca[i])
            if menor_entropia - entropia_atual <= (alpha * non_d) / 2:
                print('removendo')
                vizinhanca[i].remove(pior_vizinho)
                vizinhanca[pior_vizinho].remove(i)
            else:
                if not adicionado:
                    break

    return vizinhanca
