import numpy as np
from sklearn.model_selection import KFold
from utils import Distribuicao, Grafo
from greedy import greedy_algorithm_meu

def evaluate_algorithm(true_graph, estimated_graph):
    """
    Avalia a precisão da vizinhança estimada comparada com a verdadeira.
    """
    correct_edges = 0
    for node, neighbors in true_graph.items():
        correct_edges += len(neighbors.intersection(estimated_graph.get(node, set())))
    total_edges = sum(len(neigh) for neigh in true_graph.values())
    return correct_edges / total_edges if total_edges > 0 else 0

def cross_validate_non_d(dist, epsilon_values, k=5):
    """
    Executa validação cruzada para encontrar o melhor epsilon (non_d) para o algoritmo Greedy.
    """
    kf = KFold(n_splits=k)
    best_epsilon = None
    best_score = -np.inf

    for epsilon in epsilon_values:
        scores = []

        for train_index, val_index in kf.split(dist.amostras):
            train_samples = dist.amostras[train_index]
            val_samples = dist.amostras[val_index]

            train_dist = Distribuicao(tipo="grid", num_amostras=len(train_samples))
            train_dist.amostras = train_samples
            estimated_graph = greedy_algorithm_meu(train_dist, epsilon)

            true_graph = {i: {i+1} for i in range(dist.tamanho - 1)}
            score = evaluate_algorithm(true_graph, estimated_graph)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_epsilon = epsilon
    
    return best_epsilon, best_score
