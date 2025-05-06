from parampacmap import ParamPaCMAP, paramrep_weight_schedule, paramrep_const_schedule
from ..base import Method
from typing import Optional
import torch


class ParamPaCMAP(ParamPaCMAP, Method):
    def __init__(self,
                 n_components: int = 2,
                 n_neighbors: int = 10,
                 n_FP: int = 20,
                 n_MN: int = 5,
                 distance: str = "euclidean",
                 optim_type: str = "Adam",
                 lr: float = 1e-3,
                 lr_schedule: Optional[bool] = None,
                 apply_pca: bool = True,
                 apply_scale: Optional[str] = None,
                 model_dict: Optional[dict] = {"backbone": "ANN", "layer_size": [100, 100, 100]},
                 intermediate_snapshots: Optional[list] = [],
                 loss_weight: Optional[list] = [1, 1, 1],
                 batch_size: int = 1024,
                 data_reshape: Optional[list] = None,
                 num_epochs: int = 450,
                 verbose: bool = False,
                 weight_schedule=paramrep_weight_schedule,
                 const_schedule=paramrep_const_schedule,
                 num_workers: int = 1,
                 dtype: torch.dtype = torch.float32,
                 embedding_init: str = "pca",
                 seed: Optional[int] = None,
                 save_pairs: bool = False,
                 ):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            n_FP=n_FP,
            n_MN=n_MN,
            distance=distance,
            optim_type=optim_type,
            lr=lr,
            lr_schedule=lr_schedule,
            apply_pca=apply_pca,
            apply_scale=apply_scale,
            model_dict=model_dict,
            intermediate_snapshots=intermediate_snapshots,
            loss_weight=loss_weight,
            batch_size=batch_size,
            data_reshape=data_reshape,
            num_epochs=num_epochs,
            verbose=verbose,
            weight_schedule=weight_schedule,
            const_schedule=const_schedule,
            num_workers=num_workers,
            dtype=dtype,
            embedding_init=embedding_init,
            seed=seed,
            save_pairs=save_pairs,
        )

    def __str__(self):
        return f'ParamPaCMAP(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()