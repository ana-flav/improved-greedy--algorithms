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
    x_marginal = joint_xy.sum(axis=1, keepdims=True)
    return entropy(joint_xy.flatten(), base=2) - entropy(x_marginal.flatten(), base=2)


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
                    if len(vizinhanca[v]) > 0:
                        valores_vizinhanca = np.mean(
                            dist.amostras[:, list(vizinhanca[v])], axis=1
                        )
                    else:
                        valores_vizinhanca = dist.amostras[:, vizinho]

                    delta_n = entropia_atual - cond_entropy(
                        dist.amostras[:, v], valores_vizinhanca
                    )

                    if delta_n >= non_d / 2 and delta_n > melhor_delta:
                        melhor_delta = delta_n
                        melhor_vizinho = vizinho

            if melhor_vizinho is None:
                break

            vizinhanca[v].add(melhor_vizinho)
            entropia_atual -= melhor_delta

    return vizinhanca
