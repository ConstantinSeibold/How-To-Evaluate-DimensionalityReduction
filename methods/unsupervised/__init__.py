from .PCA import PCA, IncrementalPCA, KernelPCA
from .MDS import MDS
from .Isomap import Isomap
from .HNNE import HNNE
from .PaCMAP import PaCMAP
from .LocalMAP import LocalMAP
# from .ParamRepulsor import ParamPaCMAP
from .FactorAnalysis import FactorAnalysis
from .FastICA import FastICA
from .LatentDirichletAllocation import LatentDirichletAllocation
from .NMF import NMF
from .TruncatedSVD import TruncatedSVD
from .LocallyLinearEmbedding import LocallyLinearEmbedding
from .TSNE import TSNE
from .UMAP import UMAP
from .SNEkhorn import SNEkhorn
from .Trimap import TRIMAP
from .PHATE import PHATE
from .Random import GaussianRandomProjection, SparseRandomProjection
from .SpectralEmbedding import SpectralEmbedding


__all__ = [
    "PCA",
    "KernelPCA",
    "IncrementalPCA",
    "MDS",
    "Isomap",
    "HNNE",
    "PaCMAP",
    "LocalMAP",
    # "ParamPaCMAP",        # daemonic processes are not allowed to have children
    "FactorAnalysis",
    "FastICA",
    "LatentDirichletAllocation",
    "NMF",
    "TruncatedSVD",
    # "LinearDiscriminantAnalysis",     # requires labels
    "LocallyLinearEmbedding",
    "TSNE",
    "UMAP",
    # "SNEkhorn",           # package error
    "TRIMAP",
    "PHATE",
    "GaussianRandomProjection",
    "SparseRandomProjection",
    "SpectralEmbedding"
]