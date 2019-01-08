from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


class SharedNearestNeighbor:

    def __init__(self,n_neighbors = 7, eps = 5, min_samples = 5,n_jobs = 1):
        self.neigh =  NearestNeighbors(n_neighbors = n_neighbors, n_jobs = n_jobs)
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors

    def fit(self,X):
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        return self.dbscan.fit(self.similarity_matrix)
    
    def fit_predict(self,X):
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph*graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0,0]
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data

        self.dbscan = DBSCAN(eps = max_similarity - self.eps, min_samples = self.min_samples,metric = "precomputed", n_jobs = self.n_jobs)
        return self.dbscan.fit_predict(self.similarity_matrix)
        
        
