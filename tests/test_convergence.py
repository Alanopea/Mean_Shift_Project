import numpy as np
from mean_shift_project.mean_shift import MeanShift

def test_convergence_stops():
    X = np.random.rand(50, 2)

    ms = MeanShift(bandwidth=5.0, tol=1e-2, max_iter=100)
    ms.fit(X)

    assert ms.centroids is not None
    assert len(ms.centroids) > 0