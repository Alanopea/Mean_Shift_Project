Algorithm design — Mean Shift
=============================

In this chapter we present the design and mathematical foundations of the Mean
Shift algorithm used in this project. Scope: intuition, notation, pseudocode,
complexity, and usage examples with illustrations.

Overview
--------

Mean Shift is a non-parametric clustering algorithm that seeks density modes by
iteratively shifting points toward the local center of mass within a specified
window (the bandwidth).

Notation and equations
----------------------

Let X = {x_i}_{i=1..n} denote a set of points in R^d. For a point x we define the
neighborhood N(x) = {x_i : ||x_i - x|| < h}, where h is the bandwidth (radius).
The "flat kernel" (uniform weights) version updates x to:

.. math::
   x' = \frac{1}{|N(x)|}\sum_{x_i\in N(x)} x_i

The mean-shift vector can be written as the difference between the new and old
positions:

.. math::
   m(x) = x' - x

Pseudocode
----------

.. code-block:: text

   MeanShift algorithm(X, h, max_iter, tol):
       centroids = X.copy()
       for t in range(max_iter):
           prev = centroids.copy()
           for i, c in enumerate(centroids):
               neighbors = {x for x in X if ||x - c|| < h}
               if neighbors:
                   centroids[i] = mean(neighbors)
           if ||centroids - prev|| < tol:
               break
       return prune_centroids(centroids, h)

Complexity and practical notes
------------------------------

- Naive cost: O(n^2) per iteration for a straightforward implementation (pairwise
  distance computations); performance can be improved using spatial structures
  (KDTree) or approximate neighbor search.
- The choice of bandwidth strongly influences results: small h => more clusters;
  large h => fewer clusters.
- Convergence depends on tolerance and the maximum number of iterations.

Illustration
------------

.. figure:: _static/mean_shift_diagram.svg
   :alt: example Mean Shift diagram
   :align: center

   Example Mean Shift step: points (dots) and centroid shift (arrow).

Example usage
-------------

The following example demonstrates a simple run of the algorithm.

.. literalinclude:: ../docs/examples/mean_shift_example.py
   :language: python

References
----------

- Comaniciu, D. and Meer, P. (2002). Mean shift: A robust approach toward feature
  space analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence.
