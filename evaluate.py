import yaml
import multiprocessing as mp
from time import perf_counter


class Evaluator:
    def __init__(
            self,
            config_dict: dict | None = None,
            config_file: str | None = None,
            multiprocessing_level: str = 'datasets',
            num_processes: int = 1,
    ):
        if (config_dict is None) and (config_file is None):
            raise ValueError(f"Both `config_dict` and `config_file` are None, please specify a configuration for the "
                             f"Evaluator")

        if (config_dict is not None) and (config_file is not None):
            raise ValueError(f"Both `config_dict` and `config_file` are specified, please use only one configuration "
                             f"method")

        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = yaml.safe_load(config_file)

        assert "datasets" in self.config, f'No datasets found in configuration'
        assert "methods" in self.config, f'No methods found in configuration'
        assert "metrics" in self.config, f'No metrics found in configuration'

        # dataset validation
        for dataset in self.config['datasets']:
            assert hasattr(dataset, 'data'), f'{dataset} does not have a `data` attribute'

        for method in self.config['methods']:
            assert hasattr(method, 'fit_transform') and callable(method.fit_transform), \
                f'{method} does not have a `fit_transform()` method'
        for metric in self.config['metrics']:
            assert hasattr(metric, '__call__') and callable(metric.__call__), \
                f'{metric} does not have a `__call__()` method'

        match multiprocessing_level.lower():
            case "datasets":
                self.tasks = [{
                    data: {
                        'methods': self.config['methods'],
                        'metrics': self.config['metrics'],
                    }}
                    for data in self.config['datasets']
                ]

            case "methods":
                self.tasks = [{
                    data: {
                        'methods': [method],
                        'metrics': self.config['metrics'],
                    }}
                    for method in self.config['methods']
                    for data in self.config['datasets']
                ]

            case "metrics":
                self.tasks = [{
                    data: {
                        'methods': [method],
                        'metrics': [metric],
                    }}
                    for metric in self.config['metrics']
                    for method in self.config['methods']
                    for data in self.config['datasets']
                ]

            case _:
                raise AttributeError(f'`mp_level` set to {multiprocessing_level},'
                                     f'must be either "datasets" or "methods"')

        self.num_processes = num_processes

    @staticmethod
    def _task_eval(task_spec: dict):
        results = {}

        dataset = list(task_spec.keys())[0]
        results[dataset] = {}

        methods = task_spec[dataset]['methods']
        metrics = task_spec[dataset]['metrics']

        for method in methods:
            start_t = perf_counter()
            transformed = method.fit_transform(dataset.data)
            duration = perf_counter() - start_t
            results[dataset][method] = {}
            results[dataset][method]['Time'] = duration

            for metric in metrics:
                score = metric(transformed, dataset.data)
                results[dataset][method][metric] = score

        return results

    def run(self) -> dict:
        with mp.Pool(self.num_processes) as pool:
            result_list = pool.map(self._task_eval, self.tasks)

        final_results = {}
        for task_result in result_list:
            for dataset, method_scores in task_result.items():
                if dataset not in final_results:
                    final_results[dataset] = {}
                for method, metric_scores in method_scores.items():
                    final_results[dataset][method] = metric_scores

        return final_results


if __name__ == '__main__':
    from methods.unsupervised import PCA, MDS, Isomap, HNNE, PaCMAP, LocalMAP, FactorAnalysis, FastICA, LatentDirichletAllocation,\
        NMF, TruncatedSVD, KernelPCA, IncrementalPCA, LocallyLinearEmbedding, TRIMAP, PHATE, GaussianRandomProjection,\
        SparseRandomProjection, SpectralEmbedding
    from metrics.unsupervised import Trustworthiness, MRRE
    from datasets.supervised import Iris, Wine, Linnerud, Breast_cancer, Blobs
    import ujson as json

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
    print(json.dumps(res))
