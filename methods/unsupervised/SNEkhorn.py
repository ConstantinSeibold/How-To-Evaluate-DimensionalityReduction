from snekhorn import SNEkhorn
from ..base import Method


class SNEkhorn(SNEkhorn, Method):
    def __init__(self,
                 perp,
                 n_components=2,
                 student_kernel=False,  # True for tSNEkhorn
                 optimizer='Adam',
                 lr=1.0,
                 init='pca',
                 tol=1e-4,
                 max_iter=100,
                 lr_sea=1e-1,
                 max_iter_sea=500,
                 tol_sea=1e-3,
                 square_parametrization=False,
                 eps=1.0,  # Regularization for Sinkhorn
                 init_sinkhorn=None,
                 max_iter_sinkhorn=5,
                 tol_sinkhorn=1e-5,
                 verbose=True,
                 tolog=False
                 ):
        super().__init__(
            perp=perp,
            output_dim=n_components,
            student_kernel=student_kernel,
            optimizer=optimizer,
            lr=lr,
            init=init,
            tol=tol,
            max_iter=max_iter,
            lr_sea=lr_sea,
            max_iter_sea=max_iter_sea,
            tol_sea=tol_sea,
            square_parametrization=square_parametrization,
            eps=eps,
            init_sinkhorn=init_sinkhorn,
            max_iter_sinkhorn=max_iter_sinkhorn,
            tol_sinkhorn=tol_sinkhorn,
            verbose=verbose,
            tolog=tolog,
        )

    def __str__(self):
        return f'SNEkhorn(n_components={self.output_dim})'

    def __repr__(self):
        return self.__str__()