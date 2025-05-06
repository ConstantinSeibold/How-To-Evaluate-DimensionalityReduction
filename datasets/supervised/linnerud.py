from ..base import Dataset
from sklearn.datasets import load_linnerud


class Linnerud(Dataset):
    def __init__(self):
        linnerud = load_linnerud()
        self._data = linnerud.data
        self._labels = linnerud.target

    def __repr__(self):
        return 'Linnerud'
