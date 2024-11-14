from collections import Counter, defaultdict
import math
import numpy as np

def formata(listao):
    pares = []
    for linha in listao[0]:
        pares.append([l[linha] for l in listao])
    return pares

def empirical_cond_entropy(X, Y=[]):
    if len(Y) == 0:
        count = Counter(X)
        n = len(X)

        prob = [count[val] / n for val in count]
        return -sum(p * math.log2(p) for p in prob if p > 0)

    X = np.array(X)

    if isinstance(Y, list) and isinstance(Y[0], (list, tuple)):
        Y = np.array(Y)
    else:
        Y = np.array(Y).reshape(-1, 1)

    y_to_x = defaultdict(list)
    for i in range(len(X)):
        y_val = tuple(Y[i]) if Y.shape[1] > 1 else Y[i][0]
        y_to_x[y_val].append(X[i])

    n_samples = len(X)
    conditional_entropy = 0

    for y_val, x_vals in y_to_x.items():
        p_y = len(x_vals) / n_samples

        x_counts = defaultdict(int)
        for x in x_vals:
            x_counts[x] += 1

        entropy_xy = 0
        for x_count in x_counts.values():
            p_x_given_y = x_count / len(x_vals)
            entropy_xy -= p_x_given_y * np.log2(p_x_given_y)

        conditional_entropy += p_y * entropy_xy

    return conditional_entropy
