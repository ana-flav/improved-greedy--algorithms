import numpy as np
from greedy import greedy_algorithm_meu
from teste import cross_validate_non_d
from utils import Distribuicao, Grafo
import matplotlib.pyplot as plt


# Parâmetros e configuração
dist_d = Distribuicao(tipo="grid", num_amostras=10000)
mean = np.mean(dist_d.amostras, axis=1)

# Visualizar a distribuição dos dados
# import matplotlib.pyplot as plt
# plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()

# Lista de valores de epsilon para testar
epsilon_values = [0.00001, 0.00005, 0.0001, 0.001, 0.005]

# Realizar a validação cruzada para encontrar o melhor epsilon
# best_epsilon, best_score = cross_validate_non_d(dist_d, epsilon_values, k=5)
# print(f"Melhor epsilon: {best_epsilon} com score médio: {best_score}")

# Executar o algoritmo com o melhor epsilon encontrado
vizinhanca = greedy_algorithm_meu(dist_d, 0.0001)
mrf = Grafo.get_instance_from_vizinhanca(vizinhanca)
print("Vizinhança encontrada:", mrf.__dict__)

