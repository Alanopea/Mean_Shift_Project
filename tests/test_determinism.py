import numpy as np
from mean_shift_project.mean_shift import MeanShift

def test_deterministic_output():
    X = np.random.RandomState(42).rand(100, 2)

    ms1 = MeanShift(bandwidth=1.5)
    ms2 = MeanShift(bandwidth=1.5)

    ms1.fit(X)
    ms2.fit(X)

    c1 = np.sort(ms1.centroids, axis=0)
    c2 = np.sort(ms2.centroids, axis=0)

    assert np.allclose(c1, c2, atol=1e-4)
