from abc import ABC
from .base import Metric
from sklearn.manifold import trustworthiness
import numpy.typing as npt
import numpy as np


class Trustworthiness(Metric, ABC):
    def __call__(
            self,
            X: npt.NDArray[float],
            X_embedded: npt.NDArray[float],
            n_neighbors: int = 5,
            metric: str = 'euclidean',
            *args,
            **kwargs
    ) -> float:
        return trustworthiness(
            X=X,
            X_embedded=X_embedded,
            n_neighbors=n_neighbors,
            metric=metric,
        )

    def __repr__(self):
        return 'Trustworthiness'
