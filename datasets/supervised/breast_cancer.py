from ..base import Dataset
from sklearn.datasets import load_breast_cancer


class Breast_cancer(Dataset):
    def __init__(self):
        breast_cancer = load_breast_cancer()
        self._data = breast_cancer.data
        self._labels = breast_cancer.target

    def __repr__(self):
        return 'Breast_Cancer'
