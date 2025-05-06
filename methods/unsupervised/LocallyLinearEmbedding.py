from sklearn.manifold import LocallyLinearEmbedding
from ..base import Method


class LocallyLinearEmbedding(LocallyLinearEmbedding, Method):
    def __init__(self,
                 n_neighbors=5,
                 n_components=2,
                 reg=1e-3,
                 eigen_solver="auto",
                 tol=1e-6,
                 max_iter=100,
                 method="standard",
                 hessian_tol=1e-4,
                 modified_tol=1e-12,
                 neighbors_algorithm="auto",
                 random_state=None,
                 n_jobs=None,
                 ):
        super().__init__(
            n_neighbors=n_neighbors,
            n_components=n_components,
            reg=reg,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            method=method,
            hessian_tol=hessian_tol,
            modified_tol=modified_tol,
            neighbors_algorithm=neighbors_algorithm,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def __str__(self):
        return f'LocallyLinearEmbedding(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()