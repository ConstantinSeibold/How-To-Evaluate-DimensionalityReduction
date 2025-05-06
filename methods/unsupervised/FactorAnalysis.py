from sklearn.decomposition import FactorAnalysis
from ..base import Method


class FactorAnalysis(FactorAnalysis, Method):
    def __init__(self,
                 n_components=None,
                 tol=1e-2,
                 copy=True,
                 max_iter=1000,
                 noise_variance_init=None,
                 svd_method="randomized",
                 iterated_power=3,
                 rotation=None,
                 random_state=0,
                 ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            copy=copy,
            max_iter=max_iter,
            noise_variance_init=noise_variance_init,
            svd_method=svd_method,
            iterated_power=iterated_power,
            rotation=rotation,
            random_state=random_state,
        )

    def __str__(self):
        return f'FA(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()