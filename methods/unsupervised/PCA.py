from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from ..base import Method


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


class KernelPCA(KernelPCA, Method):
    def __init__(self,
                 n_components=None,
                 kernel="linear",
                 gamma=None,
                 degree=3,
                 coef0=1,
                 kernel_params=None,
                 alpha=1.0,
                 fit_inverse_transform=False,
                 eigen_solver="auto",
                 tol=0,
                 max_iter=None,
                 iterated_power="auto",
                 remove_zero_eig=False,
                 random_state=None,
                 copy_X=True,
                 n_jobs=None,
                 ):
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            random_state=random_state,
            copy_X=copy_X,
            n_jobs=n_jobs,
        )

    def __str__(self):
        return f'KernelPCA(n_components={self.n_components})'


class IncrementalPCA(IncrementalPCA, Method):
    def __init__(self,
                 n_components=None,
                 whiten=False,
                 copy=True,
                 batch_size=None,
                 ):
        super().__init__(
            n_components=n_components,
            whiten=whiten,
            copy=copy,
            batch_size=batch_size,
        )

    def __str__(self):
        return f'IncrementalPCA(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()