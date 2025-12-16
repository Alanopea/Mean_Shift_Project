import numpy as np
from mean_shift_project.mean_shift import MeanShift


def example():
    X = np.array([[0.0], [0.1], [5.0]])
    ms = MeanShift(bandwidth=1.0, max_iter=100, tol=1e-4)
    ms.fit(X)
    print("Centroids:", ms.centroids)


if __name__ == "__main__":
    example()
