import numpy as np
from utils import Distribuicao
from scipy.stats import entropy


def cond_entropy(x, y):
    x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")

    x_unique = len(np.unique(x))
    y_unique = len(np.unique(y))

    joint_xy = np.histogram2d(x, y, bins=[x_unique, y_unique])[0]
    joint_xy = joint_xy / joint_xy.sum()

    # x_marginal = joint_xy.sum(axis=1)
    y_marginal = joint_xy.sum(axis=0)

    return entropy(joint_xy.flatten(), base=2) - entropy(y_marginal.flatten(), base=2)


def conditional_entropy(x, y):
    x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    joint_xy = np.histogram2d(x, y, bins=[len(x_unique), len(y_unique)])[0]
    joint_xy = joint_xy / joint_xy.sum()

    y_marginal = joint_xy.sum(axis=0)

    cond_ent = 0
    for j in range(len(y_unique)):
        if y_marginal[j] > 0:
            p_x_given_y = joint_xy[:, j]
            p_x_given_y = p_x_given_y[p_x_given_y > 0]

            if len(p_x_given_y) > 0:
                cond_ent += y_marginal[j] * entropy(p_x_given_y / y_marginal[j], base=2)

    return cond_ent


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
        entropia_atual = entropy(
            np.histogram(v_aleatorio, bins=len(np.unique(v_aleatorio)))[0], base=2
        )

        while True:
            melhor_delta = -np.inf
            melhor_vizinho = None

            for vizinho in variaveis:
                if v != vizinho and vizinho not in vizinhanca[v]:
                    valores_vizinhanca = np.mean(
                        dist.amostras[:, list(vizinhanca[v]) + [vizinho]], axis=1
                    )

                    delta_n = entropia_atual - conditional_entropy(
                        dist.amostras[:, v], valores_vizinhanca
                    )

                    if delta_n >= non_d / 2 and delta_n > melhor_delta:
                        print("cai aqui")
                        melhor_delta = delta_n
                        melhor_vizinho = vizinho

            if melhor_vizinho is None:
                print("break")
                break

            vizinhanca[v].add(melhor_vizinho)
            entropia_atual -= melhor_delta

    return vizinhanca
