==================
Mean Shift Project
==================

A robust, "from-scratch" Python implementation of the Mean Shift clustering algorithm, featuring Flat Kernel logic and automatic centroid pruning.

Description
===========

This project implements the **Mean Shift** algorithm, a non-parametric clustering technique that does not require specifying the number of clusters in advance. The implementation focuses on understanding the core mechanics of density-based clustering.

Unlike standard libraries, this implementation is built purely on NumPy to demonstrate the mathematical underpinnings of the algorithm.

Key Features
------------

* **Flat (Uniform) Kernel:** Updates centroids based on the arithmetic mean of points within the bandwidth radius.
* **Centroid Pruning:** Automatically merges duplicate centroids that converge to the same peak, ensuring a clean and accurate number of clusters.
* **Scikit-Learn API Compatibility:** Implements standard ``fit(X)`` and ``predict(X)`` methods for easy integration.
* **High Test Coverage:** Verified against Scikit-Learn results to ensure parity and logical correctness.

Algorithm Details
=================

The algorithm works in two main phases:

1.  **Shifting Phase:** Every data point iteratively moves towards the mean of its neighbors (within ``bandwidth``) until convergence.
2.  **Clustering Phase (Pruning):** Points that converge to the same mode (peak) are grouped together. Centroids closer than the ``bandwidth`` distance are merged.

Usage
=====

Here is a simple example of how to use the ``MeanShift`` class:

.. code-block:: python

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

Installation
============

You can install the package locally using pip:

.. code-block:: bash

    pip install .

Or for development (editable mode):

.. code-block:: bash

    pip install -e .

Testing & Documentation
=======================

This project uses ``tox`` for managing test environments and building documentation.

To run the test suite:

.. code-block:: bash

    tox -e py314  # Or simply 'pytest' if installed manually

To build the documentation (HTML):

.. code-block:: bash

    tox -e docs

The generated documentation will be available in ``docs/_build/html/index.html``.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.