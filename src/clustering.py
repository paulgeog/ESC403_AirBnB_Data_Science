from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


REVIEW_COLS: List[str] = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

STRUCTURAL_COLS: List[str] = [
    "price",
    "accommodates",
    "amenities_count",
]


def build_feature_matrix(
    df: pd.DataFrame,
    set_name: str = "quality",
) -> Tuple[pd.DataFrame, pd.Index]:
    if set_name == "quality":
        cols = list(REVIEW_COLS)
    elif set_name == "market":
        cols = list(REVIEW_COLS) + list(STRUCTURAL_COLS)
    else:
        raise ValueError(
            f"Unknown set_name '{set_name}'. Use 'quality' or 'market'."
        )

    X = df[cols].copy()

    # log1p-transform price to compress the heavy right tail
    if "price" in X.columns:
        X["price"] = np.log1p(X["price"])

    X = X.dropna()
    return X, X.index


def fit_pca_with_scree(
    X_std: np.ndarray,
    n_components: int | None = None,
    title: str = "PCA scree plot",
) -> Tuple["PCA", np.ndarray]:
    if n_components is None:
        n_components = X_std.shape[1]

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_std)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(var_ratio) + 1)
    ax.bar(x, var_ratio, alpha=0.7, label="individual")
    ax.plot(x, cum_var, "o-", color="black", label="cumulative")
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.show()

    return pca, var_ratio


def sweep_kmeans(
    X_std: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_std)
        sil = silhouette_score(X_std, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(
    X_std: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple["KMeans", np.ndarray]:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X_std)
    return km, labels


def fit_umap_2d(
    X_std: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X_std)



def profile_clusters_numeric(
    df: pd.DataFrame,
    labels: np.ndarray,
    cols: List[str],
    aggfunc: str | List[str] = "median",
) -> pd.DataFrame:
    if len(df) != len(labels):
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but labels has {len(labels)}."
        )

    work = df[cols].copy()
    work["_cluster"] = labels

    grouped = work.groupby("_cluster")[cols].agg(aggfunc)
    sizes = work.groupby("_cluster").size().rename("n_listings")
    if isinstance(grouped.columns, pd.MultiIndex):
        sizes.index.name = "cluster"
        return grouped.join(sizes)
    grouped["n_listings"] = sizes
    grouped.index.name = "cluster"
    return grouped


def profile_clusters_categorical(
    df: pd.DataFrame,
    labels: np.ndarray,
    cols: List[str],
    top_n: int = 3,
) -> pd.DataFrame:
    if len(df) != len(labels):
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but labels has {len(labels)}."
        )

    work = df[cols].copy()
    work["_cluster"] = labels

    out = {}
    for col in cols:
        per_cluster_top = {}
        for cluster_id, sub in work.groupby("_cluster")[col]:
            counts = sub.value_counts(dropna=False).head(top_n)
            total = len(sub)
            entries = [
                f"{val} ({cnt}, {cnt / total * 100:.0f}%)"
                for val, cnt in counts.items()
            ]
            # pad to length top_n with empty strings if cluster has fewer
            entries += [""] * (top_n - len(entries))
            per_cluster_top[cluster_id] = entries

        for rank in range(top_n):
            out[(col, f"rank_{rank + 1}")] = {
                cid: per_cluster_top[cid][rank]
                for cid in per_cluster_top
            }

    result = pd.DataFrame(out)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    result.index.name = "cluster"
    return result.sort_index()