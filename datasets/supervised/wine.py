from ..base import Dataset
from sklearn.datasets import load_wine


class Wine(Dataset):
    def __init__(self):
        wine = load_wine()
        self._data = wine.data
        self._labels = wine.target

    def __repr__(self):
        return 'Wine'
