from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np


class Metric(ABC):

    @abstractmethod
    def __call__(
            self,
            X_embedded: npt.NDArray[float],
            X: npt.NDArray[float] | None = None,
            *args,
            **kwargs
    ):
        raise NotImplementedError
