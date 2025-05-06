from hnne import HNNE
from ..base import Method

class HNNE(HNNE, Method):
    def __init__(self,
                n_components: int = 2,
                metric: str = "cosine",
                radius: float = 0.4,
                ann_threshold: int = 40000,
                preliminary_embedding: str = "pca",
                random_state = None,
                ):
        super().__init__(
            n_components=n_components,
            metric=metric,
            radius=radius,
            ann_threshold=ann_threshold,
            preliminary_embedding=preliminary_embedding,
            random_state=random_state,
        )

    def __str__(self):
        return f'HNNE(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()