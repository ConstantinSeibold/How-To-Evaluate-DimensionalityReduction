from trimap import TRIMAP
from ..base import Method


class TRIMAP(TRIMAP, Method):
    def __init__(self,
                 n_components=2,
                 n_inliers=12,
                 n_outliers=4,
                 n_random=3,
                 distance="euclidean",
                 lr=0.1,
                 n_iters=400,
                 triplets=None,
                 weights=None,
                 use_dist_matrix=False,
                 knn_tuple=None,
                 verbose=False,
                 weight_adj=None,
                 weight_temp=0.5,
                 apply_pca=True,
                 opt_method="dbd",
                 return_seq=False,
                 ):
        super().__init__(
            n_dims=n_components,
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_random=n_random,
            distance=distance,
            lr=lr,
            n_iters=n_iters,
            triplets=triplets,
            weights=weights,
            use_dist_matrix=use_dist_matrix,
            knn_tuple=knn_tuple,
            verbose=verbose,
            weight_adj=weight_adj,
            weight_temp=weight_temp,
            apply_pca=apply_pca,
            opt_method=opt_method,
            return_seq=return_seq,
        )

    def __str__(self):
        return f'TRIMAP(n_components={self.n_dims})'

    def __repr__(self):
        return self.__str__()