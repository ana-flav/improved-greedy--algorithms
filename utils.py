import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Grafo:
    def __init__(self, tipo: str = None):
        self.vertices = []
        self.adjacencia = []
        self.valores = []
        self.arestas: list[tuple] = []

        if tipo == "diamante":
            self.peso_arestas = 0.5
            self.get_grafo_diamante()
        elif tipo == "grid":
            self.get_grafo_grid()

    @staticmethod
    def __get_arestas_from_adjacencia(adjacencia):
        arestas = []
        for no, vizinhanca in adjacencia.items():
            for vizinho in vizinhanca:
                aresta = (no, vizinho)
                if aresta not in arestas:
                    arestas.append(aresta)

        return arestas

    @classmethod
    def get_instance_from_vizinhanca(cls, vizinhanca: dict):
        instance = cls()
        instance.vertices = [v for v in vizinhanca.keys()]
        instance.adjacencia = vizinhanca
        instance.arestas = cls.__get_arestas_from_adjacencia(adjacencia=vizinhanca)
        return instance

    def __get_vizinho(self, vertice, aresta) -> int:
        """__get_vizinho

        Método auxiliar que retorna o vizinho de um vértice, dada uma aresta.
        Analisa os vértices presentes na aresta e retorna o vértice que não é o dado.

        Args:
            vertice (int): Índice do vértice
            aresta (tuple): Tupla que contém dois índices de vértice (cada um é uma ponta da aresta)

        Returns:
            int: Índice do vértice vizinho
        """
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
        valores = [1 for _ in range(len(grafo.vertices))]

        for vertice in grafo.vertices:
            vizinhanca = []

            for aresta in grafo.arestas:
                # Obtém a vizinhança do vértice
                if vertice in aresta:
                    vizinhanca.append(valores[self.__get_vizinho(vertice, aresta)])

            # Calcula o 'local field' relativo àquele vértice
            somatorio_vizinhanca = np.sum(grafo.peso_arestas * np.array(vizinhanca))
            prob = 1 / (1 + np.exp(-2 * somatorio_vizinhanca))

            # Escolhe "aleatóriamente" o valor do vértice de acordo com os cálculos
            valores[vertice] = 1 if random.random() < prob else -1

            if vertice == 0 or vertice == len(grafo.vertices) - 1:
                valores[vertice] = 1 if random.random() > 0.5 else -1

        grafo.valores = valores
        return grafo

    def __get_peso_aleatorio(self) -> float:
        """__get_peso_aleatorio

        Método auxiliar que escolhe aleatoriamente peso 0.25 ou -0.25
        (valores sugeridos pelo artigo), para grafos de grid.

        Returns:
            float: Peso aleatoriamente atribuído
        """
        return 0.25 if random.random() > 0.5 else -0.25

    def get_grafo_diamante(self):
        """get_grafo_diamante

        Gera um grafo diamante
        """
        self.vertices = list(range(32))

        for v in self.vertices[1:-1]:
            self.arestas.extend([(0, v), (v, self.vertices[-1])])

        self = self._get_ising_from_diamante(self)

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
                somatorio += (peso) * self.valores[v - 1]

            if v % colunas < colunas - 1:
                self.arestas.append((v, v + 1))
                peso = self.__get_peso_aleatorio()
                somatorio += (peso) * self.valores[v + 1]

            if v >= colunas:
                self.arestas.append((v, v - colunas))
                peso = self.__get_peso_aleatorio()
                somatorio += (peso) * self.valores[v - colunas]

            if v < (linhas - 1) * colunas:
                self.arestas.append((v, v + colunas))
                peso = self.__get_peso_aleatorio()
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
        self.amostras = np.array([grafo.valores for grafo in self._grafos])
        self.tamanho = len(self._grafos[0].vertices)


# Exemplo de uso:


# means = np.mean(dist_d.amostras, axis=1)
# plt.hist(means, bins=20, density=True, alpha=0.7, color="blue")
# plt.show()
# dist_g = Distribuicao(tipo="grid", num_amostras=5000)
