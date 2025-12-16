# Mean Shift Project

A robust, "from-scratch" Python implementation of the Mean Shift clustering algorithm, featuring Flat Kernel logic and automatic centroid pruning.

## Description

This project implements the **Mean Shift** algorithm, a non-parametric clustering technique that does not require specifying the number of clusters in advance. The implementation focuses on understanding the core mechanics of density-based clustering.

Unlike standard libraries, this implementation is built purely on NumPy to demonstrate the mathematical underpinnings of the algorithm.

## Key Features

- **Flat (Uniform) Kernel:** Updates centroids based on the arithmetic mean of points within the bandwidth radius.
- **Centroid Pruning:** Automatically merges duplicate centroids that converge to the same peak, ensuring a clean and accurate number of clusters.
- **Scikit-Learn API Compatibility:** Implements standard `fit(X)` and `predict(X)` methods for easy integration.
- **High Test Coverage:** Verified against Scikit-Learn results to ensure parity and logical correctness.

## Algorithm Details

The algorithm works in two main phases:

1. **Shifting Phase:** Every data point iteratively moves towards the mean of its neighbors (within `bandwidth`) until convergence.
2. **Clustering Phase (Pruning):** Points that converge to the same mode (peak) are grouped together. Centroids closer than the `bandwidth` distance are merged.

## Installation (Quick Start)

To run the code and tests, you need to install the project dependencies and the package itself.

1. **Install Dependencies:**
   (This installs NumPy, Pandas, Scikit-Learn, and testing tools like Pytest)
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. **Install the Project:**
   ```bash
   pip install -e .
   ```

## Usage

Here is a simple example of how to use the `MeanShift` class:

```python
import numpy as np
from mean_shift_project.mean_shift import MeanShift
from sklearn.datasets import make_blobs

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7)

# 2. Initialize and fit the model
ms = MeanShift(bandwidth=1.5)
ms.fit(X)

# 3. Get results
print(f"Found {len(ms.centroids)} clusters.")
print("Centroids:", ms.centroids)

# 4. Predict new points
labels = ms.predict(X)
```

## Testing

This project uses pytest for testing. Once installation is complete (see above), simply run:

```bash
pytest
```

## Documentation

To build the documentation (HTML) using Sphinx:

```bash
tox -e docs
```

The generated documentation will be available in `docs/_build/html/index.html`.

> Note: `README.rst` was converted to `README.md` and is now used as the project long description and docs overview.
   