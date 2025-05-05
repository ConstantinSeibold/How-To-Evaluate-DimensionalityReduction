from abc import ABC
from typing import Optional, Any
import numpy.typing as npt


class Dataset(ABC):
    _data: npt.NDArray[Any]
    _labels: Optional[npt.NDArray[Any]] = None

    @property
    def data(self) -> npt.NDArray[Any]:
        return self._data

    @data.setter
    def data(self, value: npt.NDArray[Any]) -> None:
        self._data = value

    @property
    def labels(self) -> Optional[npt.NDArray[Any]]:
        return self._labels

    @labels.setter
    def labels(self, value: Optional[npt.NDArray[Any]]) -> None:
        self._labels = value
