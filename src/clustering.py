from typing import Iterable, Tuple, List, Optional
import numpy as np
import pandas as pd


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
    """Build unscaled feature matrix for clustering.
    
    set_name: "quality" -> only REVIEW_COLS
              "market" -> REVIEW_COLS + STRUCTURAL_COLS (price log1p-ed)
    """
    ...


def fit_pca_with_scree(
    X_std: np.ndarray,
    n_components: int | None = None,
    title: str = "PCA scree plot",
) -> Tuple["PCA", np.ndarray]:
    """Fit PCA and show scree plot."""
    ...


def sweep_kmeans(
    X_std: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Run K-Means for each k, return inertia + silhouette."""
    ...


def fit_kmeans(
    X_std: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple["KMeans", np.ndarray]:
    """Fit final K-Means on standardized data."""
    ...


def fit_umap_2d(
    X_std: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Project standardized data into 2D via UMAP."""
    ...


def profile_clusters_numeric(
    df: pd.DataFrame,
    labels: np.ndarray,
    cols: List[str],
    aggfunc: str | List[str] = "median",
) -> pd.DataFrame:
    """Summarize numeric columns per cluster."""
    ...


def profile_clusters_categorical(
    df: pd.DataFrame,
    labels: np.ndarray,
    cols: List[str],
    top_n: int = 3,
) -> pd.DataFrame:
    """For each categorical column, list top-N modal values per cluster."""
    ...