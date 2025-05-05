from abc import ABC, abstractmethod


class Method(ABC):

    @abstractmethod
    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, X):
        raise NotImplementedError

