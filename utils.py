import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Grafo:
    def __init__(self, tipo: str):
        self.vertices = []
        self.valores = []
        self.arestas: list[tuple] = []
        self.pesos_arestas = []

        if tipo == "diamante":
            self.get_grafo_diamante()
        elif tipo == "grid":
            self.get_grafo_grid()

    def __get_vizinho(self, vertice, aresta):
        for v in aresta:
            if v != vertice:
                return v

    def _get_ising_from_diamante(self, grafo):
        """_get_ising_from_diamante

        Método auxiliar que obtém um modelo de Ising sobre um grafo diamante.
        Realiza um procedimento de amostragem de Gibbs.

        Args:
            grafo (Grafo): o grafo diamante em questão
        """
        valores = [random.choice([1, -1]) for _ in range(len(grafo.vertices))]

        for vertice in grafo.vertices:
            vizinhanca = []

            for aresta in grafo.arestas:
                # Obtém a vizinhança do vértice
                if vertice in aresta:
                    vizinhanca.append(self.__get_vizinho(vertice, aresta))

            # Calcula o 'local field' relativo àquele vértice
            somatorio_vizinhanca = np.sum(
                grafo.pesos_arestas[vertice]
                * np.array([1 for _ in range(len(vizinhanca))])
            )
            prob = 1 / (1 + np.exp(-2 * somatorio_vizinhanca))

            # Escolhe "aleatóriamente" o valor do vértice de acordo com os cálculos
            valores[vertice] = 1 if random.random() < prob else -1

        grafo.valores = valores
        return grafo

    def __get_peso_aleatorio(self):
        return 0.25 if random.random() > 0.5 else -0.25

    def get_grafo_diamante(self):
        """get_grafo_diamante

        Gera um grafo diamante
        """
        self.vertices = list(range(16))

        for v in self.vertices[1:-1]:
            self.arestas.extend([(0, v), (v, self.vertices[-1])])
            self.pesos_arestas.extend([0.5, 0.5])

        self = self.get_ising(self)

    def get_grafo_grid(self):
        """get_grafo_grid

        Gera um grafo de malha quadriculada 6x6 já como modelo de Ising
        """
        linhas = 6
        colunas = 6

        self.vertices = list(range(linhas * colunas))
        self.valores = [random.choice([1, -1]) for _ in range(len(self.vertices))]
        valores = self.valores

        for v in self.vertices:
            somatorio = 0
            if v % colunas > 0:
                self.arestas.append((v, v - 1))
                peso = self.__get_peso_aleatorio()
                self.pesos_arestas.append(peso)
                somatorio += (peso) * self.valores[v - 1]

            if v % colunas < colunas - 1:
                self.arestas.append((v, v + 1))
                peso = self.__get_peso_aleatorio()
                self.pesos_arestas.append(peso)
                somatorio += (peso) * self.valores[v + 1]

            if v >= colunas:
                self.arestas.append((v, v - colunas))
                peso = self.__get_peso_aleatorio()
                self.pesos_arestas.append(peso)
                somatorio += (peso) * self.valores[v - colunas]

            if v < (linhas - 1) * colunas:
                self.arestas.append((v, v + colunas))
                peso = self.__get_peso_aleatorio()
                self.pesos_arestas.append(peso)
                somatorio += (peso) * self.valores[v + colunas]

            prob = 1 / (1 + np.exp(-2 * somatorio))

            valores[v] = 1 if random.random() < prob else -1

        self.valores = valores


class Distribuicao:
    """Distribuicao

    Classe responsável por gerar a distrubuição sobre a qual será gerado o modelo gráfico
    """

    def __generate_amostras(self, tipo: str, num_amostras: int) -> List[Grafo]:
        return [Grafo(tipo) for _ in range(num_amostras)]

    def __init__(self, tipo: str, num_amostras: int):
        self._grafos = self.__generate_amostras(tipo, num_amostras)
        self.amostras = [grafo.valores for grafo in self._grafos]
        self.tamanho = len(self._grafos[0].vertices)


# Exemplo de uso:
# dist_d = Distribuicao(tipo="diamante", num_amostras=5000)
# dist_g = Distribuicao(tipo="grid", num_amostras=5000)
