#!/usr/bin/env python
"""Tests for `shared_nearest_neighbors` package."""

import pytest
import numpy as np

from shared_nearest_neighbors.shared_nearest_neighbors import SNN
from sklearn.cluster.tests.common import generate_clustered_data


@pytest.fixture
def n_clusters():
    return 3


@pytest.fixture
def clustered_data(n_clusters):
    return generate_clustered_data(n_clusters=n_clusters)


@pytest.fixture
def one_dimensional_data():
    return np.array(
        [
            [1],
            [3],
            [6],
        ]
    )


def test_snn_clustering(clustered_data, n_clusters):
    snn = SNN(n_neighbors=10, eps=3, min_samples=10, metric="euclidean")
    snn.fit(clustered_data)

    labels = snn.labels_
    n_clusters_out = len(set(labels)) - int(-1 in labels)
    assert n_clusters == n_clusters_out


def test_similarity_matrix(one_dimensional_data):
    n_neighbors = 2
    snn = SNN(n_neighbors=n_neighbors, eps=0.5, min_samples=1)
    similarity_matrix = snn.neighborhood_similarity_matrix(one_dimensional_data)
    shared_neighbors = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 2]])

    dissimilarity = n_neighbors - shared_neighbors

    assert np.all(np.isclose(similarity_matrix.toarray(), dissimilarity))
