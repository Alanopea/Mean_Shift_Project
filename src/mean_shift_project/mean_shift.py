import numpy as np

class MeanShift:
    """
    A simple implementation of Mean Shift clustering algorithm.
    
    Attributes:
        bandwidth (float): The radius of the neighborhood (kernel size).
        max_iter (int): Maximum number of iterations for convergence.
        tol (float): Tolerance for convergence (stopping criterion).
        centroids (ndarray): Coordinates of cluster centers found.
    """

    def __init__(self, bandwidth=4.0, max_iter=300, tol=1e-3):
        """
        Initialize the MeanShift model.

        Parameters:
        -----------
        bandwidth : float, optional (default=4.0)
            Radius of the kernel window.
        max_iter : int, optional (default=300)
            Maximum number of iterations per point.
        tol : float, optional (default=1e-3)
            Convergence threshold. If the shift is smaller than this, the algorithm stops.
        """
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        """
        Perform clustering on the input data.

        Parameters:
        -----------
        data : array-like of shape (n_samples, n_features)
            Input data points.
        """
        # Ensure data is a float array for precise calculations
        X = np.array(data, dtype=float)
        
        # Initialize centroids with all data points
        current_centroids = np.copy(X)
        
        for iteration in range(self.max_iter):
            new_centroids = []
            max_shift = 0.0
            
            # --- Modular Step: Mean Update ---
            for i, centroid in enumerate(current_centroids):
                # Calculate Euclidean distances from the current centroid to all points
                distances = np.linalg.norm(X - centroid, axis=1)
                
                # --- Kernel Computation (Flat Kernel) ---
                # Select points within the bandwidth
                points_within_bandwidth = X[distances < self.bandwidth]
                
                # Calculate new mean position
                if len(points_within_bandwidth) > 0:
                    new_centroid = np.mean(points_within_bandwidth, axis=0)
                else:
                    # If no points in radius (orphan), stay put
                    new_centroid = centroid
                
                new_centroids.append(new_centroid)
                
                # Track the shift distance for convergence check
                shift = np.linalg.norm(new_centroid - centroid)
                if shift > max_shift:
                    max_shift = shift
            
            current_centroids = np.array(new_centroids)
            
            # --- Stopping Condition ---
            if max_shift < self.tol:
                # print(f"Converged at iteration {iteration}")
                break
        
        # --- Post-processing: Prune duplicate centroids ---
        # Rounding helps to group centroids that are extremely close
        self.centroids = np.unique(np.round(current_centroids, decimals=3), axis=0)
        
    def predict(self, data):
        """
        Predict the closest cluster each sample in data belongs to.

        Parameters:
        -----------
        data : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise Exception("Model not fitted yet. Call 'fit' first.")
            
        data = np.array(data)
        predictions = []
        
        for point in data:
            # Find the nearest centroid for each point
            distances = np.linalg.norm(self.centroids - point, axis=1)
            closest_centroid_idx = np.argmin(distances)
            predictions.append(closest_centroid_idx)
            
        return np.array(predictions)