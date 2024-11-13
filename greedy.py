from collections import Counter
import math
import numpy as np
from utils import Distribuicao
from scipy.stats import entropy


def joint_entropy(*vars):
    hist, _ = np.histogramdd(vars, bins=[np.unique(var).size for var in vars])
    hist = hist / hist.sum()
    hist = hist.flatten()

    hist = hist[hist != 0]

    return -np.sum(hist * np.log2(hist))


def empirical_cond_entropy(target: np.ndarray, conditioning_set: list[np.ndarray] = []):
    if len(conditioning_set) == 0:
        count = Counter(target)
        n = len(target)

        prob = [count[val] / n for val in count]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    ent_1 = empirical_joint_entropy(target, *conditioning_set)
    ent_2 = empirical_joint_entropy(*conditioning_set)

    return ent_1 - ent_2


def empirical_joint_entropy(*vars):
    tuples = []
    n = len(vars[0])
    for i in range(len(vars[0])):
        pairs = ()
        for j in range(len(vars)):
            pairs += (int(vars[j][i]),)
        tuples.append(pairs)

    pair_counts = Counter(tuples)

    j_e = 0
    for count in pair_counts.values():
        p_xy = count / n
        j_e -= p_xy * math.log2(p_xy)

    return j_e


def _get_tuples_from_arrays(*vars):
    tuples = []
    for i in range(len(vars[0])):
        pairs = ()
        for j in range(len(vars)):
            pairs += (int(vars[j][i]),)
        tuples.append(pairs)

    return tuples


def empirical_cond_entropy(target, cond_vars=[]):
    if len(cond_vars) == 0:
        count = Counter(target)
        n = len(target)

        prob = [count[val] / n for val in count]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    n = len(target)
    data = _get_tuples_from_arrays(target, *cond_vars)
    data_cond = _get_tuples_from_arrays(*cond_vars)

    count_1 = Counter(data)
    count_2 = Counter(data_cond)

    cond_ent = 0
    for items, count in count_1.items():
        p_1 = count / n
        p_2 = count_2[tuple(x for x in items[1:])] / n
        cond_ent -= p_1 * math.log2(p_1 / p_2)

    return cond_ent


def _encontra_menor_cond_ent(dist, x, dist_x, vizinhanca_x, variaveis):
    melhor_vizinho = None
    menor_ent = 1

    nb = [dist.amostras[:, t] for t in list(vizinhanca_x)]

    for vizinho in variaveis:
        if x != vizinho and vizinho not in list(vizinhanca_x):
            nb_c = nb.copy()
            nb_c.append(dist.amostras[:, vizinho])

            entropia = empirical_cond_entropy(dist_x, nb_c)
            if entropia <= menor_ent:
                menor_ent = entropia
                melhor_vizinho = vizinho

    return melhor_vizinho, menor_ent


def greedy_algorithm_meu(dist: Distribuicao, non_d: float):
    """greed_algorithm

    Função que implementa o algoritmo guloso base.

    Args:
        dist (Distribuicao): A distribuição sobre a qual o algoritmo será aplicado
        non_d (float): Parâmetro de não-degeneração
    """
    variaveis = list(range(dist.tamanho))
    vizinhanca = {}

    for v in variaveis:
        vizinhanca[v] = set()
        v_aleatorio = dist.amostras[:, v]
        entropia_atual = empirical_cond_entropy(v_aleatorio)

        while True and len(vizinhanca[v]) < 10:
            candidato, menor_entropia = _encontra_menor_cond_ent(
                dist, v, v_aleatorio, vizinhanca[v], variaveis
            )

            if menor_entropia > (entropia_atual - non_d / 2):
                break

            delta = entropia_atual - menor_entropia
            print(
                f"({v}, {candidato}) | entropia: {menor_entropia}, atual: {entropia_atual}, delta: {delta} \n {menor_entropia < (entropia_atual - non_d / 2)}"
            )
            vizinhanca[v].add(candidato)
            entropia_atual = entropia_atual - delta

    return vizinhanca
