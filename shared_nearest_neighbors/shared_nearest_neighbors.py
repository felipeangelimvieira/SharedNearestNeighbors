from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np


class SNN(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors=7,
        eps=5,
        min_samples=5,
        weighted=False,
        n_jobs=1,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
    ):

        if weighted and eps >= 1:
            raise ValueError(
                "For weighed SNN, please define a eps value between 0 and 1."
            )
        if eps < 0:
            raise ValueError("Eps must be positive.")

        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def fit(self, X, y=None, sample_weight=None):

        self.neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
        )

        self.neigh.fit(X)

        if self.weighted:
            graph, max_similarity = self._inner_weighted(X)
        else:
            graph, max_similarity = self._inner(X)

        # In DBSCAN, eps is an upper bound of the distance between two points.
        # In terms of similarity, it would an "lower bound" on the similarity
        # or upper bound on the difference between the max similarity value and
        # the similarity between two points
        self.dbscan = DBSCAN(
            eps=max_similarity - self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

        self.dbscan.fit(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices_ = self.dbscan.core_sample_indices_
        return self

    def _inner(self, X):
        """
        Calculates the similarity matrix of the dataset
        @param X: input data matrix of shape (n_samples,n_features)
        @return graph, an sparse similarity distance matrix, and the value of the maximum similarity
        """

        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph * graph.transpose()
        self.mask = self.similarity_matrix > self.eps

        self.similarity_matrix.data = self.n_neighbors - self.similarity_matrix.data
        return graph, self.n_neighbors

    def _inner_weighted(self, X):
        """
        Calculates the similarity matrix of the dataset based on weighed edges
        @param X: input data matrix of shape (n_samples,n_features)
        @return graph, an sparse similarity distance matrix, and the value of the maximum similarity
        """

        graph = self.neigh.kneighbors_graph(X, mode="distance")
        graph.sort_indices()

        # Strength of link
        graph.data = (
            self.n_neighbors
            - np.argsort(graph.data.reshape((-1, self.n_neighbors)) + 1)
        ).flatten()
        self.similarity_matrix = graph * graph.transpose()
        max_similarity = self.similarity_matrix[0, 0]

        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data
        self.similarity_matrix.data = self.similarity_matrix.data / max(
            self.similarity_matrix.data
        )
        print("Biggest distance: ", np.max(self.similarity_matrix.data))
        print("Mean distance:", np.mean(self.similarity_matrix.data))
        print("Median distance:", np.median(self.similarity_matrix.data))
        max_similarity = 1
        return graph, max_similarity
