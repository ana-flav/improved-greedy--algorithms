import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

class GraphPlotter:
    def __init__(self, n_rows=4, n_cols=4):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_nodes = n_rows * n_cols

    def plot_graph_structure(self, estrutura_vizinhanca, dados):
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Correlação inicial
        correlation = np.corrcoef(dados.T)
        plt.subplot(131)
        sns.heatmap(correlation, cmap='coolwarm', center=0,
                   xticklabels=range(self.n_nodes), 
                   yticklabels=range(self.n_nodes))
        plt.title('Correlação dos Dados')
        
        # Plot 2: Estrutura estimada
        adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i, vizinhos in estrutura_vizinhanca.items():
            for j in vizinhos:
                adj_matrix[i,j] = 1
                adj_matrix[j,i] = 1
                
        plt.subplot(132)
        sns.heatmap(adj_matrix, cmap='Greens',
                   xticklabels=range(self.n_nodes), 
                   yticklabels=range(self.n_nodes))
        plt.title('Estrutura Estimada')
        
        # Plot 3: Estrutura verdadeira da grade
        true_adj = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            if i % self.n_cols != 0:  # Esquerda
                true_adj[i, i-1] = true_adj[i-1, i] = 1
            if (i + 1) % self.n_cols != 0:  # Direita 
                true_adj[i, i+1] = true_adj[i+1, i] = 1
            if i - self.n_cols >= 0:  # Cima
                true_adj[i, i-self.n_cols] = true_adj[i-self.n_cols, i] = 1
            if i + self.n_cols < self.n_nodes:  # Baixo
                true_adj[i, i+self.n_cols] = true_adj[i+self.n_cols, i] = 1
                
        plt.subplot(133)
        sns.heatmap(true_adj, cmap='Greens',
                   xticklabels=range(self.n_nodes), 
                   yticklabels=range(self.n_nodes))
        plt.title('Estrutura Verdadeira')
        
        plt.tight_layout()
        plt.show()

        # Análise de precisão
        true_edges = set((i,j) for i in range(self.n_nodes) for j in range(i+1, self.n_nodes) 
                        if true_adj[i,j] == 1)
        est_edges = set((i,j) for i in range(self.n_nodes) for j in range(i+1, self.n_nodes)
                       if adj_matrix[i,j] == 1)
        
        correct = len(true_edges & est_edges)
        missed = len(true_edges - est_edges) 
        extra = len(est_edges - true_edges)
        
        print("\nEstatísticas:")
        print(f"Arestas verdadeiras: {len(true_edges)}")
        print(f"Arestas estimadas corretamente: {correct}")
        print(f"Arestas não detectadas: {missed}")
        print(f"Falsos positivos: {extra}")
        if correct + extra > 0:
            print(f"Precisão: {correct/(correct + extra):.2%}")
        print(f"Recall: {correct/len(true_edges):.2%}")