import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift as SklearnMeanShift
from mean_shift_project.mean_shift import MeanShift as MyMeanShift
import time

def run_experiment():
    print('Running Mean Shift clustering experiment...')
    # 1. Data Preparation
    # We use synthetic data for clarity
    print("1. Generowanie danych syntetycznych...")
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
    
    bandwidth = 2.5
    
    # 2. Running my implementation
    print(f"\n2. Uruchamianie Twojego algorytmu (bandwidth={bandwidth})...")
    start_time = time.time()
    my_ms = MyMeanShift(bandwidth=bandwidth)
    my_ms.fit(X)
    my_time = time.time() - start_time
    my_centers = my_ms.centroids
    my_labels = my_ms.predict(X)
    print(f"   -> Czas: {my_time:.4f}s")
    print(f"   -> Znaleziono klastrów: {len(my_centers)}")

    # 3. Running Scikit-Learn implementation
    print(f"\n3. Uruchamianie Scikit-Learn (bandwidth={bandwidth})...")
    start_time = time.time()
    sklearn_ms = SklearnMeanShift(bandwidth=bandwidth)
    sklearn_ms.fit(X)
    sklearn_time = time.time() - start_time
    sklearn_centers = sklearn_ms.cluster_centers_
    sklearn_labels = sklearn_ms.labels_
    print(f"   -> Czas: {sklearn_time:.4f}s")
    print(f"   -> Znaleziono klastrów: {len(sklearn_centers)}")

    # 4. Comparison of results
    print("\n4. Porównanie wyników:")
    
    # Sort centroids for comparison
    my_centers_sorted = my_centers[my_centers[:, 0].argsort()]
    sklearn_centers_sorted = sklearn_centers[sklearn_centers[:, 0].argsort()]
    
    if len(my_centers) == len(sklearn_centers):
        # Calculate difference (error)
        diff = np.linalg.norm(my_centers_sorted - sklearn_centers_sorted)
        print(f"   -> Difference in centroids position (Error): {diff:.6f}")
        if diff < 1e-3:
            print("   -> SUKCES: Results match closely!")
        else:
            print("   -> WARNING: Results differ!")
    else:
        print("   -> ERROR: Different number of found clusters!")

    # 5. Wizualization of results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(X[:, 0], X[:, 1], c=my_labels, cmap='viridis', s=30, alpha=0.6)
    ax1.scatter(my_centers[:, 0], my_centers[:, 1], c='red', s=200, marker='X', label='Centroidy')
    ax1.set_title(f'Twój Algorytm\nClusters: {len(my_centers)} | Time: {my_time:.4f}s')
    ax1.legend()
    
    ax2.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', s=30, alpha=0.6)
    ax2.scatter(sklearn_centers[:, 0], sklearn_centers[:, 1], c='red', s=200, marker='X', label='Centroidy')
    ax2.set_title(f'Scikit-Learn\nClusters: {len(sklearn_centers)} | Time: {sklearn_time:.4f}s')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('experiment_results.png')
    print("\nZapisano wykres do 'experiment_results.png'.")
    plt.show()

if __name__ == "__main__":
    run_experiment()
    