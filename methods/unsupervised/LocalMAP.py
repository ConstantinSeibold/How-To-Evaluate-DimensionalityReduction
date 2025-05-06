from pacmap import LocalMAP
from ..base import Method


class LocalMAP(LocalMAP, Method):
    def __init__(self,
                 n_components=2,
                 n_neighbors=10,
                 MN_ratio=0.5,
                 FP_ratio=2.0,
                 pair_neighbors=None,
                 pair_MN=None,
                 pair_FP=None,
                 distance="euclidean",
                 lr=1.0,
                 num_iters=(100, 100, 250),
                 verbose=False,
                 apply_pca=True,
                 intermediate=False,
                 intermediate_snapshots=[
                     0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450],
                 random_state=None,
                 save_tree=False,
                 low_dist_thres=10
                 ):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            pair_neighbors=pair_neighbors,
            pair_MN=pair_MN,
            pair_FP=pair_FP,
            distance=distance,
            lr=lr,
            num_iters=num_iters,
            verbose=verbose,
            apply_pca=apply_pca,
            intermediate=intermediate,
            intermediate_snapshots=intermediate_snapshots,
            random_state=random_state,
            save_tree=save_tree,
            low_dist_thres=low_dist_thres
        )

    def __str__(self):
        return f'LocalMAP(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()