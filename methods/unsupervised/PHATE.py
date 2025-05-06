from phate import PHATE
from ..base import Method


class PHATE(PHATE, Method):
    def __init__(self,
                 n_components=2,
                 knn=5,
                 decay=40,
                 n_landmark=2000,
                 t="auto",
                 gamma=1,
                 n_pca=100,
                 mds_solver="sgd",
                 knn_dist="euclidean",
                 knn_max=None,
                 mds_dist="euclidean",
                 mds="metric",
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 **kwargs
                 ):
        super().__init__(
            n_components=n_components,
            knn=knn,
            decay=decay,
            n_landmark=n_landmark,
            t=t,
            gamma=gamma,
            n_pca=n_pca,
            mds_solver=mds_solver,
            knn_dist=knn_dist,
            knn_max=knn_max,
            mds_dist=mds_dist,
            mds=mds,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

    def __str__(self):
        return f'PHATE(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()