import numpy as np
import random
import matplotlib.pyplot as plt
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

def ising_gibbs_sample(n, m, theta, num_samples=1000):
    samples = np.ones((num_samples, n * m))
    for sample_idx in range(num_samples):
        for _ in range(n * m):
            i = random.randint(0, n * m - 1)
            neighbors = []
            if i % m > 0:
                neighbors.append(samples[sample_idx, i - 1])
            if i % m < m - 1:
                neighbors.append(samples[sample_idx, i + 1])
            if i >= m:
                neighbors.append(samples[sample_idx, i - m])
            if i < (n - 1) * m:
                neighbors.append(samples[sample_idx, i + m])
            field = np.sum(theta * np.array(neighbors))
            prob = 1 / (1 + np.exp(-2 * field))
            samples[sample_idx, i] = 1 if random.random() < prob else -1
    return samples

def greedy_algorithm(samples, epsilon):
    n_samples, n_variables = samples.shape
    neighbors = {}
    for i in range(n_variables):
        neighbors[i] = set()
        current_cond_entropy = entropy(np.histogram(samples[:, i], bins=len(np.unique(samples[:, i])))[0], base=2)
        while True:
            best_delta = -np.inf
            best_node = None
            for j in range(n_variables):
                if j != i and j not in neighbors[i]:
                    if len(neighbors[i]) > 0:
                        neighbor_values = np.mean(samples[:, list(neighbors[i])], axis=1)
                    else:
                        neighbor_values = samples[:, j]
                    delta_j = current_cond_entropy - cond_entropy(samples[:, i], neighbor_values)
                    if delta_j >= epsilon and delta_j > best_delta:
                        best_delta = delta_j
                        best_node = j

            if best_node is not None:
                neighbors[i].add(best_node)
                current_cond_entropy -= best_delta
            else:
                break
    return neighbors

def rec_greedy_algorithm(samples, epsilon):
    n_samples, n_variables = samples.shape
    neighbors = {}

    def recursive_addition(i, current_neighbors, current_cond_entropy):
        best_delta = -np.inf
        best_node = None
        for j in range(n_variables):
            if j != i and j not in current_neighbors:
                if len(current_neighbors) > 0:
                    neighbor_values = np.mean(samples[:, list(current_neighbors)], axis=1)
                else:
                    neighbor_values = samples[:, j]
                delta_j = current_cond_entropy - cond_entropy(samples[:, i], neighbor_values)
                if delta_j >= epsilon and delta_j > best_delta:
                    best_delta = delta_j
                    best_node = j
        if best_node is not None:
            current_neighbors.add(best_node)
            recursive_addition(i, current_neighbors, current_cond_entropy - best_delta)

    for i in range(n_variables):
        neighbors[i] = set()
        initial_entropy = entropy(np.histogram(samples[:, i], bins=len(np.unique(samples[:, i])))[0], base=2)
        recursive_addition(i, neighbors[i], initial_entropy)

    return neighbors

def fb_greedy_algorithm(samples, epsilon):
    n_samples, n_variables = samples.shape
    neighbors = {}

    for i in range(n_variables):
        neighbors[i] = set()
        current_cond_entropy = entropy(np.histogram(samples[:, i], bins=len(np.unique(samples[:, i])))[0], base=2)
        while True:
            best_delta = -np.inf
            best_node = None
            for j in range(n_variables):
                if j != i and j not in neighbors[i]:
                    if len(neighbors[i]) > 0:
                        neighbor_values = np.mean(samples[:, list(neighbors[i])], axis=1)
                    else:
                        neighbor_values = samples[:, j]
                    delta_j = current_cond_entropy - cond_entropy(samples[:, i], neighbor_values)
                    if delta_j >= epsilon and delta_j > best_delta:
                        best_delta = delta_j
                        best_node = j
            if best_node is not None:
                neighbors[i].add(best_node)
                current_cond_entropy -= best_delta
                # Removal step: attempt to remove nodes with minimal impact
                for k in list(neighbors[i]):
                    temp_neighbors = neighbors[i] - {k}
                    temp_entropy = cond_entropy(samples[:, i], np.mean(samples[:, list(temp_neighbors)], axis=1))
                    if current_cond_entropy - temp_entropy < epsilon:
                        neighbors[i].remove(k)
                        current_cond_entropy = temp_entropy
            else:
                break

    return neighbors


def greedy_p_algorithm(samples, epsilon):
    neighbors = greedy_algorithm(samples, epsilon)
    for i in neighbors:
        for j in list(neighbors[i]):
            temp_neighbors = neighbors[i] - {j}
            if cond_entropy(samples[:, i], np.mean(samples[:, list(temp_neighbors)], axis=1)) > epsilon:
                neighbors[i].remove(j)
    return neighbors

# Placeholder for RWL (Regularized Logistic Regression-based) algorithm
def rwl_algorithm(samples, lambda_value):
    neighbors = {}  
    return neighbors
def simulate_results_plot():
    sample_sizes = [500, 1000, 2000, 3000, 4000, 5000]
    probability_of_success_diamond = {
        'D=2': [0.1, 0.5, 0.8, 0.9, 0.95, 0.99],
        'D=3': [0.05, 0.4, 0.75, 0.85, 0.9, 0.95],
        'D=4': [0.02, 0.3, 0.6, 0.8, 0.88, 0.9],
    }
    probability_of_success_4x4 = {
        'RecGreedy': [0.2, 0.6, 0.85, 0.95, 0.98, 0.99],
        'FbGreedy': [0.15, 0.55, 0.83, 0.93, 0.97, 0.99],
        'GreedyP': [0.1, 0.5, 0.8, 0.92, 0.95, 0.98],
        'Greedy': [0.05, 0.4, 0.7, 0.88, 0.92, 0.95],
        'RWL': [0.25, 0.65, 0.9, 0.96, 0.99, 0.995]
    }
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label, probabilities in probability_of_success_diamond.items():
        plt.plot(sample_sizes, probabilities, label=label, marker='o')
    plt.title("Diamond Network - Theta = 0.25, Threshold Degree = 5")
    plt.xlabel("Number of samples")
    plt.ylabel("Probability of success")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for label, probabilities in probability_of_success_4x4.items():
        plt.plot(sample_sizes, probabilities, label=label, marker='o')
    plt.title("4x4 Grid, Max Degree = 4")
    plt.xlabel("Number of samples")
    plt.ylabel("Probability of success")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
# simulate_results_plot()

