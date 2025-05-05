from sklearn.manifold import Isomap
from .base import Method


class Isomap(Isomap, Method):
    def __init__(self,
                 n_neighbors=5,
                 radius=None,
                 n_components=2,
                 eigen_solver='auto',
                 tol=0,
                 max_iter=None,
                 path_method='auto',
                 neighbors_algorithm='auto',
                 n_jobs=None,
                 metric='minkowski',
                 p=2,
                 metric_params=None
                 ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            n_components=n_components,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            n_jobs=n_jobs,
            metric=metric,
            p=p,
            metric_params=metric_params,
        )

    def __repr__(self):
        return f'Isomap(n_components={self.n_components})'
