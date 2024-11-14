from utils_entropy import empirical_cond_entropy
import numpy as np
from sample import Distribuicao

def rec_greedy(dist: Distribuicao, epsilon: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {i: set() for i in variaveis} 

    for i in variaveis:
        N_i = set() 
        iterate = True

        while iterate:
            T_i = N_i.copy() 
            last = 0
            complete = False

            while not complete:
                melhor_candidato = None
                menor_entropia = np.inf
                # aqui vem a selecao do no joota que minimiza a entropia e estara no conjunto T_i
                    if entropia_sem_j - menor_entropia >= epsilon / 2:
                        T_i.add(melhor_candidato)
                        last = melhor_candidato
                    else:
                        if last != 0:
                            N_i.add(last)
                        else:
                            iterate = False
                        complete = True
                else:
                    if last != 0:
                        N_i.add(last)
                    else:
                        iterate = False
                    complete = True

        vizinhanca[i] = N_i

    edges = {(i, j) for i in vizinhanca for j in vizinhanca[i] if i < j}
    return edges
