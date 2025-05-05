from sklearn.manifold import MDS
from .base import Method


class MDS(MDS, Method):
    def __init__(self,
                 n_components=2,
                 metric=True,
                 n_init=4,
                 max_iter=300,
                 verbose=0,
                 eps=0.001,
                 n_jobs=None,
                 random_state=None,
                 dissimilarity='euclidean',
                 normalized_stress='auto',
                 ):
        super().__init__(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            dissimilarity=dissimilarity,
            normalized_stress=normalized_stress,
        )

    def __repr__(self):
        return f'MDS(n_components={self.n_components})'
