import numpy as np
from mean_shift_project.mean_shift import MeanShift

def test_flat_kernel_mean_update():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [10.0, 10.0]
    ])

    ms = MeanShift(bandwidth=2.0, max_iter=1)
    ms.fit(X)

    expected_mean = np.mean([[0.0, 0.0], [1.0, 1.0]], axis=0)

    assert any(np.allclose(c, expected_mean, atol=1e-3) for c in ms.centroids)


def test_points_outside_bandwidth_do_not_affect_centroid():
    X = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [100.0, 100.0]
    ])

    ms = MeanShift(bandwidth=1.0, max_iter=1)
    ms.fit(X)

    expected = np.mean([[0.0, 0.0], [0.5, 0.5]], axis=0)

    assert any(np.allclose(c, expected, atol=1e-3) for c in ms.centroids)
