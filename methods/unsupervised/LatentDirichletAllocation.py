from sklearn.decomposition import LatentDirichletAllocation
from ..base import Method


class LatentDirichletAllocation(LatentDirichletAllocation, Method):
    def __init__(self,
                 n_components=10,
                 doc_topic_prior=None,
                 topic_word_prior=None,
                 learning_method="batch",
                 learning_decay=0.7,
                 learning_offset=10.0,
                 max_iter=10,
                 batch_size=128,
                 evaluate_every=-1,
                 total_samples=1e6,
                 perp_tol=1e-1,
                 mean_change_tol=1e-3,
                 max_doc_update_iter=100,
                 n_jobs=None,
                 verbose=0,
                 random_state=None,
                 ):
        super().__init__(
            n_components=n_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method=learning_method,
            learning_decay=learning_decay,
            learning_offset=learning_offset,
            max_iter=max_iter,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            total_samples=total_samples,
            perp_tol=perp_tol,
            mean_change_tol=mean_change_tol,
            max_doc_update_iter=max_doc_update_iter,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

    def __str__(self):
        return f'LatentDirichletAllocation(n_components={self.n_components})'

    def __repr__(self):
        return self.__str__()