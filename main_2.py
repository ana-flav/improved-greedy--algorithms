import numpy as np
from greedy import greedy_algorithm_meu
from utils import Distribuicao, Grafo
import matplotlib.pyplot as plt


dist_d = Distribuicao(tipo="grid", num_amostras=200)
mean = np.mean(dist_d.amostras, axis=1)

plt.hist(mean, bins=20, density=True, alpha=0.7, color="blue")
plt.show()

vizinhanca = greedy_algorithm_meu(dist_d, 0.04)
mrf = Grafo.get_instance_from_vizinhanca(vizinhanca)
print(mrf.__dict__)
