from greedy import greedy_algorithm_meu
from utils import Distribuicao, Grafo


dist_d = Distribuicao(tipo="grid", num_amostras=5000)
vizinhanca = greedy_algorithm_meu(dist_d, 0.002)
mrf = Grafo.get_instance_from_vizinhanca(vizinhanca)
print(mrf.__dict__)