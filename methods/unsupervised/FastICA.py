from sklearn.decomposition import FastICA
from ..base import Method


class FastICA(FastICA, Method):
    def __init__(self,
                 n_components=None,
                 algorithm="parallel",
                 whiten="unit-variance",
                 fun="logcosh",
                 fun_args=None,
                 max_iter=200,
                 tol=1e-4,
                 w_init=None,
                 whiten_solver="svd",
                 random_state=None,
                 ):
        super().__init__(
            n_components=n_components,
            algorithm=algorithm,
            whiten=whiten,
            fun=fun,
            fun_args=fun_args,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            whiten_solver=whiten_solver,
            random_state=random_state,
        )

    def __str__(self):
        return f'FastICA(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()