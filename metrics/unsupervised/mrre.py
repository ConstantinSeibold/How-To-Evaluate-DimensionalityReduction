from zadu.measures.mean_relative_rank_error import measure
from ..base import Metric


class MRRE(Metric):
    def __init__(self, k: int = 20):
        self.k = k

    def __call__(self,
                 orig,
                 emb,
                 k=None,
                 knn_ranking_info=None,
                 return_local=False
                 ):
        if k is None:
            k = self.k
        return measure(
            orig=orig,
            emb=emb,
            k=k,
            knn_ranking_info=knn_ranking_info,
            return_local=return_local,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'MRRE'
