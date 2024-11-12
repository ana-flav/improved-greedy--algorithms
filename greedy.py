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


def calculate_conditional_entropy_ising(
    target_var: np.ndarray, conditioning_vars: np.ndarray
) -> float:
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
        # If no conditioning variables, return marginal entropy
        hist, _ = np.histogram(target_var, bins=[-np.inf, 0, np.inf])
        hist = hist / len(target_var)
        return entropy(hist, base=2)

    # For Ising model, we need to consider the joint states
    if conditioning_vars.ndim == 1:
        conditioning_vars = conditioning_vars.reshape(-1, 1)

    # Create unique states for conditioning variables
    unique_states = np.unique(conditioning_vars, axis=0)

    cond_entropy = 0
    for state in unique_states:
        # Find samples matching this configuration
        mask = conditioning_vars.flatten() == state

        # Calculate conditional probability
        p_state = np.mean(mask)

        # Get target variable values for this configuration
        target_given_state = target_var[mask]

        # Calculate entropy for this configuration
        if len(target_given_state) > 0:
            hist, _ = np.histogram(target_given_state, bins=[-np.inf, 0, np.inf])
            hist = hist / len(target_given_state)
            if np.any(hist > 0):  
                cond_entropy += p_state * entropy(hist, base=2)

    return cond_entropy


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
        # entropia_atual = entropy(
        #     np.histogram(v_aleatorio, bins=len(np.unique(v_aleatorio)))[0], base=2
        # )

        entropia_atual = calculate_conditional_entropy_ising(v_aleatorio, np.array([]))

        while True:
            melhor_delta = -np.inf
            melhor_vizinho = None

            for vizinho in variaveis:
                if v != vizinho and vizinho not in vizinhanca[v]:
                    valores_vizinhanca = dist.amostras[:, vizinho]

                    nova_entropia = calculate_conditional_entropy_ising(
                        v_aleatorio, valores_vizinhanca
                    )

                    delta_n = entropia_atual - nova_entropia

                    if delta_n >= non_d / 2 and delta_n > melhor_delta:
                        melhor_delta = delta_n
                        melhor_vizinho = vizinho

            if melhor_vizinho is None:
                break

            vizinhanca[v].add(melhor_vizinho)
            entropia_atual -= melhor_delta

    return vizinhanca
