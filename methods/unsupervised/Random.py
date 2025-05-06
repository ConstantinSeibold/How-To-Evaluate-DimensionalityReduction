from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from ..base import Method


class GaussianRandomProjection(GaussianRandomProjection, Method):
    def __init__(self,
                 n_components="auto",
                 eps=0.1,
                 compute_inverse_components=False,
                 random_state=None,
                 ):
        super().__init__(
            n_components=n_components,
            eps=eps,
            compute_inverse_components=compute_inverse_components,
            random_state=random_state,
        )

    def __str__(self):
        return f'GaussianRandomProjection(n_components={self.n_components})'


class SparseRandomProjection(SparseRandomProjection, Method):
    def __init__(self,
                 n_components="auto",
                 density="auto",
                 eps=0.1,
                 dense_output=False,
                 compute_inverse_components=False,
                 random_state=None,
                 ):
        super().__init__(
            n_components=n_components,
            density=density,
            eps=eps,
            dense_output=dense_output,
            compute_inverse_components=compute_inverse_components,
            random_state=random_state,
        )

    def __str__(self):
        return f'SparseRandomProjection(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()