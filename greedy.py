import numpy as np
from utils import Distribuicao
from scipy.stats import entropy


def cond_entropy(target_var: np.ndarray, conditioning_vars: np.ndarray) -> float:
    """
    Calculate conditional entropy for Ising model variables.

    Args:
        samples: Array of samples
        target_var: Target variable values
        conditioning_vars: Conditioning variables values

    Returns:
        Conditional entropy value
    """
    if conditioning_vars.size == 0:
        hist, _ = np.histogram(target_var, bins=[-np.inf, 0, np.inf])
        hist = hist / len(target_var)
        return entropy(hist, base=2)

    if conditioning_vars.ndim == 1:
        conditioning_vars = conditioning_vars.reshape(-1, 1)

    unique_states = np.unique(conditioning_vars, axis=0)

    cond_entropy = 0
    for state in unique_states:
        mask = conditioning_vars.flatten() == state

        p_state = np.mean(mask)
        target_given_state = target_var[mask]

        if len(target_given_state) > 0:
            hist, _ = np.histogram(target_given_state, bins=[-np.inf, 0, np.inf])
            hist = hist / len(target_given_state)
            if np.any(hist > 0):
                cond_entropy += p_state * entropy(hist, base=2)

    return cond_entropy


def _encontra_menor_cond_ent(dist, x, dist_x, vizinhanca_x, variaveis):
    melhor_vizinho = None
    menor_ent = 1

    for vizinho in variaveis:
        if x != vizinho and vizinho not in vizinhanca_x:
            entropia = cond_entropy(dist_x, dist.amostras[:, vizinho])
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
        entropia_atual = cond_entropy(v_aleatorio, np.array([]))

        while True:
            melhor_delta = -np.inf
            melhor_vizinho = None

            talvez_melhor_vizinho, melhor_entropia = _encontra_menor_cond_ent(
                dist, v, v_aleatorio, vizinhanca[v], variaveis
            )
            delta_n = entropia_atual - melhor_entropia

            if melhor_entropia < (entropia_atual - non_d / 2):
                print(f"entropia obtida: {melhor_entropia}")
                print(
                    f"analisando {v} e {talvez_melhor_vizinho},\n delta_n -- {delta_n}"
                )
                melhor_delta = delta_n
                melhor_vizinho = talvez_melhor_vizinho

            if melhor_vizinho is None:
                break

            vizinhanca[v].add(melhor_vizinho)
            entropia_atual -= melhor_delta

    return vizinhanca
