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

    Ertoz, Levent, Michael Steinbach, and Vipin Kumar. "A new shared nearest neighbor clustering algorithm and its applications." Workshop on clustering high dimensional data and its applications at 2nd SIAM international conference on data mining. 2002.

    """

    def __init__(self,n_neighbors = 7, eps = 5, min_samples = 5, weighted = False, n_jobs = 1, algorithm = 'auto', leaf_size = 30, metric = 'minkowski', p = 2, metric_params = None):
        """
        @param n_neighbors: number of neighbors to consider when calculating the shared nearest neighbors
        @param eps: threshold on the number of neighbors
        @param min_samples: minimum number  of samples that share at least eps neighbors so that a point can be considered  a core point
        @param weighted: if True, uses weighed version o Shared Nearest Neighbors, and eps must be between 0 and 1.
        @param n_jobs: number of parallel jobs
        @param algorithm: parameter for Nearest Neighbors calculation, please see scikit-learn documentation
        @param leaf_size: parameter for Nearest Neighbors calculation, please see scikit-learn documentation
        @param metric: parameter for Nearest Neighbors, please see scikit-learn documentation
        @param p: param for Nearest Neighbors, please see scikit-learn documentation
        @param metric_params: param for Nearest Neighbors, please see scikit-learn documentation
        @return an instance of SharedNearestNeighbor class
        """
        if weighted and eps >= 1:
            raise ValueError("For weighed SNN, please define a eps value between 0 and 1.")
        if eps < 0:
            raise ValueError("Eps must be positive.")

        self.neigh =  NearestNeighbors(n_neighbors = n_neighbors, n_jobs = n_jobs, algorithm= algorithm, leaf_size= leaf_size, metric= metric, p = p, metric_params= metric_params)
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.labels_ = []
        self.components_ = []
        self.core_sample_indices_ = []
        self.weighted = weighted

        

    def fit(self,X):
        """
        @param X: array of shape (n_samples, n_features)
        @return self
        """
        if self.weighted:
            graph, max_similarity = self._inner_weighted(X)    
        else: 
            graph, max_similarity = self._inner(X)
        

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        self.dbscan.fit(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices_ = self.dbscan.core_sample_indices_
        return self

    def _inner(self,X): 
        """
        Calculates the similarity matrix of the dataset

        @param X: input data matrix of shape (n_samples,n_features)
        @return graph, an sparse similarity distance matrix, and the value of the maximum similarity

        """
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data
        return graph, max_similarity

    def _inner_weighted(self,X):
        """
        Calculates the similarity matrix of the dataset based on weighed edges

        @param X: input data matrix of shape (n_samples,n_features)
        @return graph, an sparse similarity distance matrix, and the value of the maximum similarity
        """
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X, mode = "distance")
        graph.data = np.reshape(self.n_neighbors - np.argsort(np.argsort(graph.data.reshape((-1,self.n_neighbors)))),(-1,))
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        assert(max_similarity == np.max(self.similarity_matrix))
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data
        self.similarity_matrix.data = self.similarity_matrix.data/max(self.similarity_matrix.data)
        print("Biggest distance: ", np.max(self.similarity_matrix.data))
        print("Mean distance:",np.mean(self.similarity_matrix.data))
        print("Median distance:",np.median(self.similarity_matrix.data))
        max_similarity = 1
        return graph, max_similarity

    def fit_predict(self,X):
        """
        @param X: array of shape (n_samples, n_features)
        @return y: array of shape (n_samples), containing the predicted cluster labels
        """
        if self.weighted:
            graph, max_similarity = self._inner_weighted(X)    
        else: 
            graph, max_similarity = self._inner(X)

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        y = self.dbscan.fit_predict(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices_ = self.dbscan.core_sample_indices_
        return y
        
        
