from openTSNE import TSNE
from ..base import Method


class TSNE(TSNE, Method):
    def __init__(self,
                 n_components=2,
                 perplexity=30,
                 learning_rate="auto",
                 early_exaggeration_iter=250,
                 early_exaggeration="auto",
                 n_iter=500,
                 exaggeration=None,
                 dof=1,
                 theta=0.5,
                 n_interpolation_points=3,
                 min_num_intervals=50,
                 ints_in_interval=1,
                 initialization="pca",
                 metric="euclidean",
                 metric_params=None,
                 initial_momentum=0.8,
                 final_momentum=0.8,
                 max_grad_norm=None,
                 max_step_norm=5,
                 n_jobs=1,
                 neighbors="auto",
                 negative_gradient_method="auto",
                 callbacks=None,
                 callbacks_every_iters=50,
                 random_state=None,
                 verbose=False,
                 ):
        super().__init__(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration_iter=early_exaggeration_iter,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            exaggeration=exaggeration,
            dof=dof,
            theta=theta,
            n_interpolation_points=n_interpolation_points,
            min_num_intervals=min_num_intervals,
            ints_in_interval=ints_in_interval,
            initialization=initialization,
            metric=metric,
            metric_params=metric_params,
            initial_momentum=initial_momentum,
            final_momentum=final_momentum,
            max_grad_norm=max_grad_norm,
            max_step_norm=max_step_norm,
            n_jobs=n_jobs,
            neighbors=neighbors,
            negative_gradient_method=negative_gradient_method,
            callbacks=callbacks,
            callbacks_every_iters=callbacks_every_iters,
            random_state=random_state,
            verbose=verbose,
        )

    def __str__(self):
        return f'TSNE(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()