import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift as SklearnMeanShift
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from mean_shift_project.mean_shift import MeanShift as MyMeanShift
import pandas as pd


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "_static")
os.makedirs(OUT_DIR, exist_ok=True)


def run_single(X, true_labels, bandwidth, method="mine"):
    if method == "mine":
        ms = MyMeanShift(bandwidth=bandwidth)
        start = time.time()
        ms.fit(X)
        elapsed = time.time() - start
        centers = ms.centroids
        labels = ms.predict(X)
    else:
        ms = SklearnMeanShift(bandwidth=bandwidth)
        start = time.time()
        ms.fit(X)
        elapsed = time.time() - start
        centers = ms.cluster_centers_
        labels = ms.labels_

    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)
    return {
        "centers": centers,
        "labels": labels,
        "time": elapsed,
        "ari": ari,
        "nmi": nmi,
        "n_clusters": 0 if centers is None else len(centers),
    }


def run_bandwidth_sweep(X, true_labels, bandwidths, repeats=5):
    rows = []
    for bw in bandwidths:
        for r in range(repeats):
            res_mine = run_single(X, true_labels, bw, method="mine")
            res_sk = run_single(X, true_labels, bw, method="sklearn")
            rows.append({
                "method": "mine",
                "bandwidth": bw,
                "n_clusters": res_mine["n_clusters"],
                "ari": res_mine["ari"],
                "nmi": res_mine["nmi"],
                "time": res_mine["time"],
            })
            rows.append({
                "method": "sklearn",
                "bandwidth": bw,
                "n_clusters": res_sk["n_clusters"],
                "ari": res_sk["ari"],
                "nmi": res_sk["nmi"],
                "time": res_sk["time"],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "experiment_table.csv")
    df.to_csv(csv_path, index=False)

    # Plotting: number of clusters vs bandwidth
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, grp in df.groupby("method"):
        means = grp.groupby("bandwidth")["n_clusters"].mean()
        ax.plot(means.index, means.values, marker='o', label=method)
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("mean number of clusters")
    ax.set_title("Clusters vs Bandwidth")
    ax.legend()
    bw_path = os.path.join(OUT_DIR, "bandwidth_sweep.png")
    fig.tight_layout()
    fig.savefig(bw_path)

    # Plotting: ARI vs bandwidth
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, grp in df.groupby("method"):
        means = grp.groupby("bandwidth")["ari"].mean()
        ax.plot(means.index, means.values, marker='o', label=method)
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("mean ARI")
    ax.set_title("ARI vs Bandwidth")
    ax.legend()
    ari_path = os.path.join(OUT_DIR, "bandwidth_ari.png")
    fig.tight_layout()
    fig.savefig(ari_path)

    print(f"Saved bandwidth sweep plots and CSV to {OUT_DIR}")


def run_scaling_test(n_samples_list, centers=4, cluster_std=0.8, bandwidth=2.5):
    rows = []
    for n in n_samples_list:
        X, labels = make_blobs(n_samples=n, centers=centers, cluster_std=cluster_std, random_state=42)
        start = time.time()
        my_ms = MyMeanShift(bandwidth=bandwidth)
        my_ms.fit(X)
        elapsed = time.time() - start
        rows.append({"n_samples": n, "time": elapsed})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "scaling_table.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_samples"], df["time"], marker='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n_samples (log scale)')
    ax.set_ylabel('time (s, log scale)')
    ax.set_title('Runtime Scaling of MyMeanShift')
    scale_path = os.path.join(OUT_DIR, "runtime_scaling.png")
    fig.tight_layout()
    fig.savefig(scale_path)

    print(f"Saved runtime scaling plot and CSV to {OUT_DIR}")


def run_experiment():
    print('Running Mean Shift clustering experiments...')
    X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

    # Baseline comparison (single run)
    bw = 2.5
    res_mine = run_single(X, y, bw, method="mine")
    res_sk = run_single(X, y, bw, method="sklearn")

    # Save the visual comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=res_mine["labels"], cmap='viridis', s=30, alpha=0.6)
    if res_mine["centers"] is not None and len(res_mine["centers"])>0:
        ax1.scatter(res_mine["centers"][:, 0], res_mine["centers"][:, 1], c='red', s=200, marker='X')
    ax1.set_title(f'MyMeanShift\nClusters: {res_mine["n_clusters"]} | Time: {res_mine["time"]:.4f}s | ARI: {res_mine["ari"]:.3f}')

    ax2.scatter(X[:, 0], X[:, 1], c=res_sk["labels"], cmap='viridis', s=30, alpha=0.6)
    if res_sk["centers"] is not None and len(res_sk["centers"])>0:
        ax2.scatter(res_sk["centers"][:, 0], res_sk["centers"][:, 1], c='red', s=200, marker='X')
    ax2.set_title(f'Scikit-Learn\nClusters: {res_sk["n_clusters"]} | Time: {res_sk["time"]:.4f}s | ARI: {res_sk["ari"]:.3f}')

    out_path = os.path.join(OUT_DIR, 'experiment_results.png')
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved baseline comparison plot to {out_path}")

    # Bandwidth sweep
    bandwidths = np.linspace(0.5, 4.0, 8)
    run_bandwidth_sweep(X, y, bandwidths, repeats=5)

    # Scaling test
    run_scaling_test([100, 250, 500, 1000, 2000], centers=4, cluster_std=0.8, bandwidth=bw)


if __name__ == "__main__":
    run_experiment()
    