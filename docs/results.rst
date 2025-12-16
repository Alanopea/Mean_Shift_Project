.. _results:

==================
Experiment Results
==================

This section provides a detailed, quantitative comparison between the custom **MeanShift** implementation in this project and the reference implementation from **Scikit-Learn**.

Methodology
===========
Experiments are run on synthetic datasets generated with ``sklearn.datasets.make_blobs``. Unless specified otherwise:

* **Samples:** 500
* **Centers:** 4
* **Cluster Std:** 0.8
* **Random Seed:** 42
* **Bandwidth (baseline):** 2.5

Metrics
=======
We report the following metrics to evaluate correctness and performance:

- **Number of clusters**: number of unique centroids after pruning.
- **Centroid L2 error**: L2 norm between matched centroids of both implementations (when counts match).
- **ARI (Adjusted Rand Index)**: measure of clustering agreement with the ground truth labels.
- **NMI (Normalized Mutual Information)**: another clustering similarity metric.
- **Runtime**: elapsed time for the algorithm to converge.

Visual and Quantitative Results
===============================

Baseline comparison (single run)
--------------------------------

.. image:: /_static/experiment_results.png
   :alt: Baseline clustering comparison
   :align: center
   :width: 100%

Bandwidth sensitivity
---------------------

We sweep bandwidth in a range and report mean number of detected clusters and mean ARI.

.. image:: /_static/bandwidth_sweep.png
   :alt: Clusters vs bandwidth
   :align: center
   :width: 80%

.. image:: /_static/bandwidth_ari.png
   :alt: ARI vs bandwidth
   :align: center
   :width: 80%

Runtime scaling
---------------

We run the algorithm for increasing dataset sizes and measure runtime (log-log plot).

.. image:: /_static/runtime_scaling.png
   :alt: Runtime scaling
   :align: center
   :width: 80%

Data and reproducibility
------------------------

Detailed results (per-run): :download:`experiment_table.csv </_static/experiment_table.csv>` and :download:`scaling_table.csv </_static/scaling_table.csv>` are available in the documentation assets.

Conclusions
===========

- **Correctness:** The custom implementation produces centroids and clusterings that largely agree with Scikit-Learn for reasonable bandwidth values (ARI typically close to 1.0 for the synthetic dataset used).
- **Bandwidth sensitivity:** Choice of bandwidth strongly affects the number of clusters; the sweep plot is useful to select a stable bandwidth.
- **Performance:** The NumPy implementation is clear and educational but has worse scaling for large datasets (see runtime plot); consider KDTree or approximate neighbor search for production use.

Notes
-----
All experiments are deterministic given the random seeds described above. Results and images are saved under ``docs/_static`` for inclusion in the documentation.