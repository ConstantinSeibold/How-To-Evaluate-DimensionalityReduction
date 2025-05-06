from sklearn.decomposition import NMF
from ..base import Method


class NMF(NMF, Method):
    def __init__(self,
                 n_components="auto",
                 init=None,
                 solver="cd",
                 beta_loss="frobenius",
                 tol=1e-4,
                 max_iter=200,
                 random_state=None,
                 alpha_W=0.0,
                 alpha_H="same",
                 l1_ratio=0.0,
                 verbose=0,
                 shuffle=False,
                 ):
        super().__init__(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle,
        )

    def __str__(self):
        return f'NMF(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()