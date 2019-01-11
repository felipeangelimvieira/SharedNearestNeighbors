# SharedNearestNeighbors
A Shared Nearest Neighbors clustering implementation. This code is basically a wrapper of sklearn DBSCAN, implementing the neighborhood similarity as a metric.

## Examples

The implementation follows the syntax of scikit-learn clustering classes.
See example notebook for more information.

```python
from SNN import SharedNearestNeighbor as SNN

X = np.random.rand(100,2)
snn = SNN(n_neighbors = 10,eps = 2, min_samples = 2)
y_pred = snn.fit_predict(X)
```

## References

Ertöz, Levent, Michael Steinbach, and Vipin Kumar. "Finding clusters of different sizes, shapes, and densities in noisy, high dimensional data." Proceedings of the 2003 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2003.
