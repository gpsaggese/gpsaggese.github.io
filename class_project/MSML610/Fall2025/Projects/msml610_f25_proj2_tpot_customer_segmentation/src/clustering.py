import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

def evaluate(X, labels):
    return dict(
        silhouette=float(silhouette_score(X, labels)),
        calinski_harabasz=float(calinski_harabasz_score(X, labels)),
        davies_bouldin=float(davies_bouldin_score(X, labels)),
    )

def search_models(X, k_min=2, k_max=10, random_state=42):
    rows = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km_labels = km.fit_predict(X)
        rows.append({"algo": "KMeans", "k": k, **evaluate(X, km_labels)})

        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gm_labels = gm.fit_predict(X)
        rows.append({"algo": "GMM", "k": k, **evaluate(X, gm_labels)})

    return pd.DataFrame(rows).sort_values(
        ["silhouette", "calinski_harabasz"], ascending=[False, False]
    )

def fit_best(X, best_row):
    algo = best_row["algo"]
    k = int(best_row["k"])
    if algo == "KMeans":
        model = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labels = model.labels_
    else:
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X)
        labels = model.predict(X)
    return model, labels

def pca_plot(X, labels, path_png):
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    plt.figure()
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(X2d[idx, 0], X2d[idx, 1], s=10, label=f"Cluster {c}")
    plt.title("Clusters in PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close()
