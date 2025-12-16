import numpy as np


class MeanShift:
    """
    Mean Shift clustering implementation using a Flat (Uniform) Kernel.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Radius of the flat kernel. Points within this distance contribute 
        equally to the mean shift; points outside are ignored.
    
    max_iter : int, default=300
        Maximum number of iterations for the centroid shift loop.
    
    tol : float, default=1e-3
        Tolerance for convergence. The algorithm stops if the shift in centroids
        is less than this value.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [0.1], [5.0]])
    >>> ms = MeanShift(bandwidth=1.0)
    >>> ms.fit(X)
    >>> ms.centroids is not None
    True
    """

    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Perform clustering on data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        """
        current_centroids = np.array(X, copy=True)
        
        for _ in range(self.max_iter):
            prev_centroids = np.array(current_centroids, copy=True)
            
            for i, centroid in enumerate(current_centroids):
                distances = np.linalg.norm(X - centroid, axis=1)
                
                # Flat Kernel: Select points strictly within the bandwidth
                points_within_bandwidth = X[distances < self.bandwidth]
                
                # Update centroid to the arithmetic mean of neighbors
                if len(points_within_bandwidth) > 0:
                    current_centroids[i] = np.mean(points_within_bandwidth, axis=0)
            
            # Check for convergence
            shift = np.linalg.norm(current_centroids - prev_centroids)
            if shift < self.tol:
                break
        
        self.centroids = self._prune_centroids(current_centroids)

    def _prune_centroids(self, points):
        """ 
        Filter converged points to remove duplicates within the bandwidth radius.

        Parameters
        ----------
        points : np.array
            The converged points after the shifting phase.

        Returns
        -------
        np.array
            Unique centroids.
        """
        unique_centroids = []
        
        for point in points:
            if not unique_centroids:
                unique_centroids.append(point)
                continue
            
            centroids_array = np.array(unique_centroids)
            dists = np.linalg.norm(centroids_array - point, axis=1)
            
            if np.all(dists > self.bandwidth):
                unique_centroids.append(point)
        
        return np.array(unique_centroids)

    def predict(self, X):
        """
        Predict the closest cluster centroid for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : np.array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise Exception("Model not fitted yet. Call 'fit' first.")
            
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)