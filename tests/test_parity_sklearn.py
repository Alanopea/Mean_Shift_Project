import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift as SklearnMeanShift
from mean_shift_project.mean_shift import MeanShift

def test_parity_with_sklearn():
    X, _ = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.7,
        random_state=0
    )

    bandwidth = 1.5

    my_ms = MeanShift(bandwidth=bandwidth)
    my_ms.fit(X)

    sk_ms = SklearnMeanShift(bandwidth=bandwidth)
    sk_ms.fit(X)

    assert len(my_ms.centroids) == len(sk_ms.cluster_centers_)

    my_sorted = my_ms.centroids[np.argsort(my_ms.centroids[:, 0])]
    sk_sorted = sk_ms.cluster_centers_[np.argsort(sk_ms.cluster_centers_[:, 0])]

    error = np.linalg.norm(my_sorted - sk_sorted)
    assert error < 1e-1
