#!/usr/bin/env python
"""Tests for `shared_nearest_neighbors` package."""

import pytest
import numpy as np

from shared_nearest_neighbors.shared_nearest_neighbors import SNN


@pytest.fixture
def simple_data():
    return np.array([[1], [3], [5], [10], [11]])


def test_similarity_matrix(simple_data):
    snn = SNN(n_neighbors=2, eps=0.5, min_samples=1, weighted=True)
    snn.fit(simple_data)
