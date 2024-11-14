from utils_entropy import empirical_cond_entropy
import numpy as np
from sample import Distribuicao

def rec_greedy(dist: Distribuicao, epsilon: float):
    variaveis = list(range(dist.tamanho))
    vizinhanca = {i: set() for i in variaveis}  # Inicializar a vizinhança estimada N̂(i)

    for i in variaveis:
        N_i = set()  # N̂(i) - Conjunto de vizinhança estimada para o nó i
        iterate = True

        while iterate:
            T_i = N_i.copy()  # T̂(i) começa com o valor atual de N̂(i)
            last = None
            complete = False

            while not complete:
                melhor_candidato = None
                menor_entropia = np.inf

                for k in variaveis:
                    if k != i and k not in T_i:
                
                        entropia_atual = empirical_cond_entropy(
                            dist.amostras[:, i], [dist.amostras[:, v] for v in T_i] + [dist.amostras[:, k]]
                        )
                       
                        if entropia_atual < menor_entropia:
                            menor_entropia = entropia_atual
                            melhor_candidato = k

                if melhor_candidato is not None:
                 
                    entropia_sem_j = empirical_cond_entropy(
                        dist.amostras[:, i], [dist.amostras[:, v] for v in T_i]
                    )
                   
                    if entropia_sem_j - menor_entropia >= epsilon / 2:
                        T_i.add(melhor_candidato)
                        last = melhor_candidato
                    else:
                        if last is not None:
                            N_i.add(last)
                        else:
                            iterate = False
                        complete = True
                else:
                    if last is not None:
                        N_i.add(last)
                    else:
                        iterate = False
                    complete = True


            vizinhanca[i] = N_i

    edges = {(i, j) for i in vizinhanca for j in vizinhanca[i] if i < j}
    return edges
