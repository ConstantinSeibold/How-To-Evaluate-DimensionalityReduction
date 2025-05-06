from evaluate import Evaluator
from methods.unsupervised import PCA, MDS, Isomap, HNNE, PaCMAP, LocalMAP, FactorAnalysis, FastICA, LatentDirichletAllocation,\
        NMF, TruncatedSVD, KernelPCA, IncrementalPCA, LocallyLinearEmbedding, TRIMAP, PHATE, GaussianRandomProjection,\
        SparseRandomProjection, SpectralEmbedding
from metrics.unsupervised import Trustworthiness, MRRE
from datasets.supervised import Iris, Wine, Linnerud, Breast_cancer, Blobs

import ujson as json
import pandas as pd

if __name__ == '__main__':
    conf = {
        'datasets': [
            Iris(),
            Wine(),
            Linnerud(),
            Breast_cancer(),
            Blobs(n_samples=1000, n_features=10, center_box=(20.0, 20.0)),
            ],
        'methods': [
            PCA(n_components=2),
            MDS(n_components=2),
            Isomap(n_components=2),
            HNNE(n_components=2),
            PaCMAP(n_components=2),
            LocalMAP(n_components=2),
            FactorAnalysis(n_components=2),
            FastICA(n_components=2),
            LatentDirichletAllocation(n_components=2),
            NMF(n_components=2),
            TruncatedSVD(n_components=2),
            IncrementalPCA(n_components=2),
            KernelPCA(n_components=2),
            LocallyLinearEmbedding(n_components=2),
            TRIMAP(n_components=2),
            PHATE(n_components=2),
            GaussianRandomProjection(n_components=2),
            SparseRandomProjection(n_components=2),
            SpectralEmbedding(n_components=2),
        ],
        'metrics': [
            Trustworthiness(),
            MRRE(k=10),
            ],
    }

    ev = Evaluator(config_dict=conf, multiprocessing_level='datasets', num_processes=5)
    res = ev.run()