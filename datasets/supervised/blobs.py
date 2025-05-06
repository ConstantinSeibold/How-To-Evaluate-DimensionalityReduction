from ..base import Dataset
from sklearn.datasets import make_blobs


class Blobs(Dataset):
    def __init__(self,
                 n_samples=100,
                 n_features=2,
                 centers=None,
                 cluster_std=1.0,
                 center_box=(-10.0, 10.0),
                 shuffle=True,
                 random_state=None,
                 return_centers=False,
                 ):
        self._data, self._labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
            random_state=random_state,
            return_centers=return_centers,
        )

    def __repr__(self):
        return 'Blobs'
