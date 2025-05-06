from sklearn.manifold import SpectralEmbedding
from ..base import Method


class SpectralEmbedding(SpectralEmbedding, Method):
    def __init__(self,
                 n_components=2,
                 affinity="nearest_neighbors",
                 gamma=None,
                 random_state=None,
                 eigen_solver=None,
                 eigen_tol="auto",
                 n_neighbors=None,
                 n_jobs=None,
                 ):
        super().__init__(
            n_components=n_components,
            affinity=affinity,
            gamma=gamma,
            random_state=random_state,
            eigen_solver=eigen_solver,
            eigen_tol=eigen_tol,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )

    def __str__(self):
        return f'SpectralEmbedding(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()