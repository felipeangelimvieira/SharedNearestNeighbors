from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np


class SharedNearestNeighbor:
    """
    Shared Nearest Neighbor clustering  algorithm for finding clusters or different sizes, shapes and densities in
    noisy, high-dimensional datasets. 

    The algorithm can be seen as a variation of DBSCAN which uses neighborhood similairty as a metric.
    It does not have a hard time detecting clusters of different densities as DBSCAN, and keeps its advantages.
    
    References:

    Ertöz, L., Steinbach, M., & Kumar, V. (2003, May). Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data. In Proceedings of the 2003 SIAM international conference on data mining (pp. 47-58). Society for Industrial and Applied Mathematics.
    """

    def __init__(self,n_neighbors = 7, eps = 5, min_samples = 5,n_jobs = 1):
        """
        @param n_neighbors: number of neighbors to consider when calculating the shared nearest neighbors
        @param eps: threshold on the number of neighbors
        @param min_samples: minimum number  of samples that share at least eps neighbors so that a point can be considered  a core point
        @param n_jobs: number of parallel jobs
        @return an instance of SharedNearestNeighbor class
        """
        self.neigh =  NearestNeighbors(n_neighbors = n_neighbors, n_jobs = n_jobs)
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors

    def fit(self,X):
        """
        @param X: array of shape (n_samples, n_features)
        @return y: array of shape (n_samples), containing the predicted cluster labels
        """

        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        return self.dbscan.fit(self.similarity_matrix)
    
    def fit_predict(self,X):
        """
        @param X: array of shape (n_samples, n_features)
        @return y: array of shape (n_samples), containing the predicted cluster labels
        """
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        return self.dbscan.fit_predict(self.similarity_matrix)
        
        
