# Mean Shift - Algorithm Design & Theoretical Foundations

## 1. Theoretical Foundations
The Mean Shift algorithm is a non-parametric clustering technique that does not require prior knowledge of the number of clusters. It works by iteratively shifting data points towards the mode (highest density) of the data distribution.

* **Kernel Density Estimation (KDE):** The core idea is based on KDE, where each data point generates a "kernel" (e.g., a Gaussian or flat window). The algorithm treats the data points as a probability density function.
* **Bandwidth Selection:** The bandwidth (often denoted as $h$ or radius) is the most critical parameter. It determines the size of the kernel/window.
    * Small bandwidth: Results in many small, fragmented clusters.
    * Large bandwidth: Merges distinct clusters into a few large ones.
* **Convergence Logic:** The algorithm proceeds iteratively:
    1.  For each centroid, identify neighbors within the `bandwidth`.
    2.  Calculate the **mean vector** of these neighbors.
    3.  Shift the centroid to this new mean.
    4.  Repeat until the shift (Euclidean distance) is smaller than the `tolerance` or `max_iter` is reached.

## 2. Input/Output & Parameter Design

### Input
* **Data (X):** A 2D array-like structure (NumPy array) of shape `(n_samples, n_features)`.
    * Example: A set of 300 points in 2D space `(300, 2)`.

### Output
* **Cluster Centers:** An array of coordinates representing the peaks of the density (modes). Shape: `(n_clusters, n_features)`.
* **Labels:** An array assigning each input point to the nearest cluster center. Shape: `(n_samples,)`.

### Parameters
* `bandwidth` (float): The radius of the neighborhood (kernel size). Default: `4.0` (must be estimated or provided).
* `max_iter` (int): The maximum number of iterations to prevent infinite loops (safety break). Default: `300`.
* `tolerance` (float): The threshold for convergence. If the shift in centroid position is less than this value, the algorithm stops. Default: `1e-3` (0.001).

## 3. Dataset & Preprocessing Pipeline
For the purpose of verifying algorithm correctness and parity with `scikit-learn`, we will use **Synthetic Numerical Data**:
* **Source:** `sklearn.datasets.make_blobs`
* **Preprocessing:**
    * **Normalization:** Not strictly required for synthetic blobs but recommended for real-world data (StandardScaler).
    * **Type Casting:** All input data will be converted to `float64` to ensure precision during mean calculation.

*Future Extension (Optional):* The algorithm can be applied to **Image Data** (pixels (R, G, B) as 3D points) for image segmentation tasks.

## 4. Comparison Criteria vs. scikit-learn
To pass the project requirements (30 pts for Correctness), we will compare:
1.  **Cluster Centers:** Calculate the Euclidean distance between centroids found by our implementation and `sklearn.cluster.MeanShift`.
2.  **Number of Clusters:** Verify if `len(my_centroids) == len(sklearn_centroids)`.
3.  **Labels:** Visual comparison using scatter plots (colors matching clusters).
4.  **Convergence:** Ensure our loop terminates effectively under the same bandwidth conditions.