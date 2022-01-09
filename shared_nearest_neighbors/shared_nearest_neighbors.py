import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix, spdiags

def snn_dissimilarity_func(graph : csr_matrix, n_neighbors : int, *args, **kwargs) -> csr_matrix:
    """Default SNN dissimilarity function

    Computes the dissimilarity between two points in terms of shared nearest neighbors

    Args:
        graph (scipy.sparse.csr_matrix): sparse matrix with dimensions (n_samples, n_samples),
         where the element ij represents the distance between the point i and j 
        n_neighbors (int): number of neighbors in the k-neighborhood search
    """ 

    graph.data[graph.data > 0] = 1
    n_samples = graph.shape[0]

    # Add the point as its own neighbor
    graph += spdiags(np.ones(n_samples), diags=0, m=n_samples, n=n_samples)
    matrix = graph * graph.transpose()
    matrix.sort_indices()

    # The lower the "closer"
    matrix.data = n_neighbors - matrix.data

    return matrix


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
        dissimilarity_func=snn_dissimilarity_func,
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

        dissimilarity_func: Callable, optional
            A function that receives two inputs: scipy.sparse.csr_matrix with the k-neighbors distance and the n_neighbors attribute;
            and returns another csr_matrix

        Attributes
        ----------
        
        neigh : sklearn.neighbors.NearestNeighbors 
        
        dbscan : sklearn.cluster.DBSCAN

        labels_ : ndarray of shape (n_samples)
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.
            
        components_ : ndarray of shape (n_core_samples, n_features)

        Copy of each core sample found by training.

        core_samples_indices_ : ndarray of shape (n_core_samples,)
            Indices of core samples.

        dissimilarity_matrix : scipy.sparse.csr_matrix 
            containing the dissimilarity between points

        References
        ----------

        Ertöz, L., Steinbach, M., & Kumar, V. (2003, May). Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data. In Proceedings of the 2003 SIAM international conference on data mining (pp. 47-58). Society for Industrial and Applied Mathematics.
        Ertoz, Levent, Michael Steinbach, and Vipin Kumar. "A new shared nearest neighbor clustering algorithm and its applications." Workshop on clustering high dimensional data and its applications at 2nd SIAM international conference on data mining. 2002.


        """

        if eps <= 0:
            raise ValueError("Eps must be positive.")
        if eps >= n_neighbors and dissimilarity_func == snn_dissimilarity_func:
            raise  ValueError("Eps must be smaller than n_neighbors.")

        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.dissimilarity_func = dissimilarity_func
        self.neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
        )

        # Reasoning behind eps=self.n_neighbors - self.eps:
        # In DBSCAN, eps is an upper bound of the distance between two points.
        # In terms of similarity, it would an "lower bound" on the similarity
        # or, once again, upper bound on the difference between the max similarity value and
        # the similarity between two points
        self.dbscan = DBSCAN(
            eps=self.n_neighbors - self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )

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

        self.dissimilarity_matrix = self.neighborhood_dissimilarity_matrix(X)

        self.dbscan.fit(self.dissimilarity_matrix, sample_weight=sample_weight)

        return self

    @property
    def labels_(self):
        return self.dbscan.labels_
    
    @property
    def components_(self):
        return self.dbscan.components_

    @property
    def core_sample_indices_(self):
        return self.dbscan.core_sample_indices_


    def neighborhood_dissimilarity_matrix(self, X) -> csr_matrix:
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
        graph = self.neigh.kneighbors_graph(X, mode="distance")
        dissimilarity_matrix = self.dissimilarity_func(graph, self.n_neighbors)
        return dissimilarity_matrix