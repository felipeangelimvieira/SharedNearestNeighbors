from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.sparse import csr_matrix


class SNN(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors=7,
        eps=5,
        min_samples=5,
        algorithm="auto",
        leaf_size=30,
        metric="euclidean",
        p=None,
        metric_params=None,
        n_jobs=None,
    ):
        """Shared Nearest Neighbor clustering  algorithm for finding clusters or different sizes, shapes and densities in
        noisy, high-dimensional datasets.


        The algorithm can be seen as a variation of DBSCAN which uses neighborhood similairty as a metric.
        It does not have a hard time detecting clusters of different densities as DBSCAN, and keeps its advantages.


        Parameters
        ----------
        n_neighbors : int, optional
            The number of neighbors to construct the neighborhood graph, including the point itself. By default 7
        eps : int, optional
            The minimum number of neighbors two points have to share in order to be
            connected by an edge in the neighborhood graph. This value has to be smaller
            than n_neighbors. By default 5
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself, by default 5

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details., by default "auto"
        leaf_size : int, optional
            [description], by default 30
        metric : str, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a :term:`Glossary <sparse graph>`, in which
            case only "nonzero" elements may be considered neighbors for DBSCAN.
            Default to "euclidean"
        p : int, optional
            The power of the Minkowski metric to be used to calculate distance
            between points. If None, then ``p=2`` (equivalent to the Euclidean
            distance).
        metric_params : [type], optional
            Additional keyword arguments for the metric function., by default None
        n_jobs : int, optional
            The number of parallel jobs to run.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details. Default None.

        Attributes
        ----------


        References
        ----------

        Ertöz, L., Steinbach, M., & Kumar, V. (2003, May). Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data. In Proceedings of the 2003 SIAM international conference on data mining (pp. 47-58). Society for Industrial and Applied Mathematics.
        Ertoz, Levent, Michael Steinbach, and Vipin Kumar. "A new shared nearest neighbor clustering algorithm and its applications." Workshop on clustering high dimensional data and its applications at 2nd SIAM international conference on data mining. 2002.


        """

        if eps <= 0:
            raise ValueError("Eps must be positive.")

        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def fit(self, X, y=None, sample_weight=None):
        """Perform SNN clustering from features or distance matrix

        First calls NearestNeighbors to construct the neighborhood graph considering the params
        n_neighbors, n_jobs, algorithm, leaf_size, metric, p, metric_params

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """

        self.neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
        )

        self.similarity_matrix = self.neighborhood_similarity_matrix(X)

        # In DBSCAN, eps is an upper bound of the distance between two points.
        # In terms of similarity, it would an "lower bound" on the similarity
        # or upper bound on the difference between the max similarity value and
        # the similarity between two points
        self.dbscan = DBSCAN(
            eps=self.n_neighbors - self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=self.n_jobs,
            sample_weight=sample_weight,
        )

        self.dbscan.fit(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices_ = self.dbscan.core_sample_indices_
        return self

    def neighborhood_similarity_matrix(self, X) -> csr_matrix:
        """Neighborhood similarity matrix

        Computes the sparse neighborhood similarity matrix

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        Returns
        -------
        csr_matrix
            Sparse matrix of shape (n_samples, n_samples)
        """

        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X, mode="connectivity")
        similarity_matrix = graph * graph.transpose()
        similarity_matrix.sort_indices()
        similarity_matrix.data = self.n_neighbors - self.similarity_matrix.data
        return similarity_matrix
