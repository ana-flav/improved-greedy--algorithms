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

def empirical_conditional_entropy(target, cond_set):
    """
    Calcula a entropia condicional empírica de uma variável aleatória `target` 
    dado um conjunto de variáveis aleatórias `cond_set`.
    
    Parâmetros:
    - target (np.ndarray): Array 1D representando a variável alvo.
    - cond_set (list of np.ndarray): Lista de arrays 1D representando as variáveis condicionantes.
    
    Retorna:
    - float: Entropia condicional H(target | cond_set)
    """
    # Transforma cond_set em uma matriz, onde cada linha é uma realização das variáveis em cond_set
    cond_matrix = np.vstack(cond_set).T
    
    # Junta target com cond_set para formar a distribuição conjunta
    joint_data = np.column_stack((target, cond_matrix))
    
    # Contar as frequências para a distribuição conjunta (target, cond_set)
    joint_counts = Counter(map(tuple, joint_data))
    joint_total = sum(joint_counts.values())
    
    # Entropia conjunta H(target, cond_set)
    H_joint = -sum((count / joint_total) * np.log2(count / joint_total) for count in joint_counts.values())
    
    # Contar as frequências para a distribuição marginal cond_set
    cond_counts = Counter(map(tuple, cond_matrix))
    cond_total = sum(cond_counts.values())
    
    # Entropia marginal H(cond_set)
    H_cond = -sum((count / cond_total) * np.log2(count / cond_total) for count in cond_counts.values())
    
    # Entropia condicional H(target | cond_set)
    H_conditional = H_joint - H_cond
    return H_conditional


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


def greedy_anaflavia(dist: Distribuicao, non_d: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {}

    for i in variaveis:
        vizinhanca[i] = set()
        entropia_atual = empirical_cond_entropy(dist.amostras[:, i])

        while True:
            melhor_delta = -np.inf
            melhor_candidato = None

            for j in variaveis:
                if j != i and j not in vizinhanca[i]:
                    entropia_j = empirical_conditional_entropy(
                        dist.amostras[:, i],
                        [dist.amostras[:, v] for v in vizinhanca[i]]
                        + [dist.amostras[:, j]],
                    )

                    delta = entropia_atual - entropia_j
                    if delta >= non_d / 2 and delta > melhor_delta:
                        melhor_delta = delta
                        melhor_candidato = j

            if melhor_candidato is not None:
                vizinhanca[i].add(melhor_candidato)
                entropia_atual -= melhor_delta
            else:
                break

    return vizinhanca
