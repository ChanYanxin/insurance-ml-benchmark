from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centroids: pd.DataFrame
    silhouette: float


def cluster_numeric_profiles(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> ClusterResult:
    """Cluster age/BMI/children profiles after standardization."""
    cols = ["age", "bmi", "children"]
    values = df[cols].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(scaled)
    centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=cols)
    return ClusterResult(labels=labels, centroids=centroids, silhouette=float(silhouette_score(scaled, labels)))
