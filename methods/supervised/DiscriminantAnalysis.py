from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..base import Method


class LinearDiscriminantAnalysis(LinearDiscriminantAnalysis, Method):
    def __init__(self,
                 solver="svd",
                 shrinkage=None,
                 priors=None,
                 n_components=None,
                 store_covariance=False,
                 tol=1e-4,
                 covariance_estimator=None,
                 ):
        super().__init__(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
            covariance_estimator=covariance_estimator,
        )

    def __str__(self):
        return f'LinearDiscriminantAnalysis(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()