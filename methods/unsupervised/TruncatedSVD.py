from sklearn.decomposition import TruncatedSVD
from ..base import Method


class TruncatedSVD(TruncatedSVD, Method):
    def __init__(self,
                 n_components=2,
                 algorithm="randomized",
                 n_iter=5,
                 n_oversamples=10,
                 power_iteration_normalizer="auto",
                 random_state=None,
                 tol=0.0,
                 ):
        super().__init__(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
            tol=tol,
        )

    def __str__(self):
        return f'TruncatedSVD(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()