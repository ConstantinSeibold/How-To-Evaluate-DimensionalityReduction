from sklearn.decomposition import PCA
from .base import Method


class PCA(PCA, Method):
    def __init__(self,
                 n_components=None,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',
                 n_oversamples=10,
                 power_iteration_normalizer='auto',
                 random_state=None,
                 ):
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )
