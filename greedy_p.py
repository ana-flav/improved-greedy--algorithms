from utils_entropy import empirical_cond_entropy
import numpy as np
from sample import Distribuicao
from greedy import greedy

def greedyP(dist: Distribuicao, epsilon: float):

    vizinhanca = greedy(dist, epsilon)

    for i in vizinhanca:
        N_i = vizinhanca[i].copy() 
        for j in N_i:
            entropia_completa = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i]]
            )
            entropia_sem_j = empirical_cond_entropy(
                dist.amostras[:, i], [dist.amostras[:, v] for v in vizinhanca[i] if v != j]
            )


            gamma_j = entropia_sem_j - entropia_completa


            if gamma_j <= epsilon / 2:
                vizinhanca[i].remove(j)
  
    edges = {(i, j) for i in vizinhanca for j in vizinhanca[i] if i < j}
    return edges
