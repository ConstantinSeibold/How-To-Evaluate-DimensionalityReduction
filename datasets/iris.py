from .base import Dataset
from sklearn.datasets import load_iris


class Iris(Dataset):
    def __init__(self):
        iris = load_iris()
        self._data = iris.data
        self._labels = iris.target

    def __repr__(self):
        return 'Iris'
