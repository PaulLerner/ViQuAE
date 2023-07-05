"""
Optimize hyperparameters of the ir.search pipeline.

Usage:
hp.py <type> <config> [--train=<dataset> --k=<k> --disable_caching --cleanup_cache_files --metrics=<path> --test=<dataset>]

Positional arguments:
    1. <type>      'bm25' to optimize BM25 hyperparameters
    2. <config>    Path to the JSON configuration file (passed as kwargs)
    
Options:
    --train=<dataset>       Name of the train dataset
    --k=<k>                 Hyperparameter to search for the k nearest neighbors [default: 100].
    --disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
    --cleanup_cache_files   Clean up all cache files in the dataset cache directory, 
                            excepted the currently used one, see Dataset.cleanup_cache_files()
                            Useful to avoid saturating disk storage (caches are only deleted when exiting with --disable_caching)
    --metrics=<path>        Path to save the metrics in JSON and TeX format (only applicable with --test)
    --test=<dataset>        Name of the test dataset
"""
import warnings

from docopt import docopt
import json
from collections import Counter
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
from datasets import load_from_disk, set_caching_enabled
import ranx

import optuna

from .metrics import find_relevant_batch
from .search import Searcher, format_qrels_indices

# TODO https://github.com/castorini/pyserini/blob/master/docs/usage-interactive-search.md#how-do-i-configure-search

class Objective:
    """
    Callable objective compatible with optuna.
    
    Parameters
    ----------
    dataset: Dataset
        Used to optimize hyperparameters
    do_cache_relevant: bool,
        Whether to cache relevant results are not.
    metric_for_best_model: str, optional
        Metric used to evaluate the model. Defaults to "mrr@{self.searcher.k}" (e.g. "mrr@100")
    eval_dataset: Dataset, optional
        Used to evaluated after optimization
    map_kwargs, fn_kwargs: dict, optional
        Passed to self.dataset.map
    cleanup_cache_files: bool, optional
        Clean up all cache files in the dataset cache directory, 
        excepted the currently used one, see Dataset.cleanup_cache_files()
    **kwargs:
        passed to Searcher
    """
    def __init__(self, dataset, do_cache_relevant, metric_for_best_model=None, eval_dataset=None,
                 map_kwargs={}, fn_kwargs={}, cleanup_cache_files=False, **kwargs):
        self.dataset = dataset
        self.searcher = Searcher(**kwargs)
        # HACK: sleep until elasticsearch is good to go
        time.sleep(60)
        if metric_for_best_model is None:
            self.metric_for_best_model = f"mrr@{self.searcher.k}"
        else:
            self.metric_for_best_model = metric_for_best_model
        self.eval_dataset = eval_dataset
        self.map_kwargs = map_kwargs
        self.fn_kwargs = fn_kwargs
        self.do_cache_relevant = do_cache_relevant
        self.cleanup_cache_files = cleanup_cache_files

    def __call__(self, trial):
        """
        Tries value suggested by trial, 
        runs self.dataset through self.searcher.fuse_and_compute_metrics 
        and evaluates the results
        """
        pass

    def evaluate(self, best_params):
        """
        Should evaluate self.eval_dataset with best_params

        Parameters
        ----------
        best_params: dict

        Returns
        -------
        report: ranx.Report
        """
        pass

    def cache_relevant(self, batch, do_copy=False):
        """
        Caches relevant passages w.r.t. union of all search results.
        """
        all_indices = self.searcher.union_results(batch)
        relevant_batch = deepcopy(batch['provenance_indices']) if do_copy else batch['provenance_indices']
        provenance_indices = find_relevant_batch(all_indices, batch['output'], self.searcher.reference_kb,
                                                 reference_key=self.searcher.reference_key, relevant_batch=relevant_batch)
        str_indices_batch, non_empty_scores = format_qrels_indices(provenance_indices)
        self.searcher.qrels.add_multi(
            q_ids=batch['id'],
            doc_ids=str_indices_batch,
            scores=non_empty_scores
        )
        return batch
    
    def cache_relevant_dataset(self, do_copy=False):
        """Maps self.dataset through self.cache_relevant and handles KB"""
        self.dataset.map(self.cache_relevant, batched=True, fn_kwargs=dict(do_copy=do_copy), **self.map_kwargs)
        self.keep_reference_kb = self.searcher.reference_kb
        # so that subsequent calls to searcher.fuse_and_compute_metrics will not call find_relevant_batch
        self.searcher.reference_kb = None


class BM25Objective(Objective):
    """
    Used to optimize hyperparameters of BM25.
    
    Parameters
    ----------
    hyp_hyp: dict, optional
        Contains hyper-hyperparameters: the bounds and steps of grid search.
        Defaults to ``{"bounds": (0, 1.1), "step": 0.1}``
    settings: dict, optional
        settings that will be filled with tried values before being fed to self.es_client.indices.put_settings
        Defaults to ``{'similarity': {'karpukhin': {'b': 0.75, 'k1': 1.2}}}``
    """
    def __init__(self, *args, hyp_hyp=None, settings=None, **kwargs):                 
        super().__init__(*args, **kwargs)
        # default parameters
        if hyp_hyp is None:
            self.hyp_hyp = {
                "b": {
                    "bounds": (0, 1),
                    "step": 0.1
                },
                "k1": {
                    "bounds": (0, 3),
                    "step": 0.1
                }
            }
        else:
            self.hyp_hyp = hyp_hyp
        if settings is None:
            self.settings = {'similarity': {'karpukhin': {'b': 0.75, 'k1': 1.2}}}
        else:
            self.settings = settings

        # check that there is a single ES index + save ES client and ES client’s name
        self.index_name = None
        for kb in self.searcher.kbs.values():
            for index_name, index in kb.indexes.items():
                if index.es:
                    assert self.index_name is None, f"Expected a single ES index, got {self.index_name} and {index_name}"
                    self.index_name = index_name
                    self.es_client = kb.es_client
                    es_index = kb.dataset._indexes[self.index_name]
                    self.es_index_name = es_index.es_index_name

        assert self.index_name is not None, "Did not find an ES index"

    def __call__(self, trial):
        settings = self.settings

        # suggest hyperparameters
        b = trial.suggest_float("b", *self.hyp_hyp["b"]["bounds"])
        k1 = trial.suggest_float("k1", *self.hyp_hyp["k1"]["bounds"])
        for parameters in settings['similarity'].values():
            parameters['b'] = b
            parameters['k1'] = k1
        # close index, update its settings then open it
        self.es_client.indices.close(self.es_index_name)
        self.es_client.indices.put_settings(settings, self.es_index_name)
        self.es_client.indices.open(self.es_index_name)

        self.dataset.map(self.searcher, fn_kwargs=self.fn_kwargs, batched=True, **self.map_kwargs)
        if self.searcher.do_fusion:            
            raise NotImplementedError()  
        run = self.searcher.runs[self.index_name]
        metric = ranx.evaluate(self.searcher.qrels, run, self.metric_for_best_model)
        return metric

    def evaluate(self, best_params):
        # reset to erase qrels and runs of the validation set
        self.searcher.qrels = ranx.Qrels()
        self.searcher.runs = dict()
        for kb in self.searcher.kbs.values():
            for index_name, index in kb.indexes.items():
                run = ranx.Run()
                run.name = index_name
                self.searcher.runs[index_name] = run
        if self.searcher.do_fusion:
            raise NotImplementedError()
        
        settings = self.settings

        for parameters in settings['similarity'].values():
            parameters.update(best_params)
        # close index, update its settings then open it
        self.es_client.indices.close(self.es_index_name)
        self.es_client.indices.put_settings(settings, self.es_index_name)
        self.es_client.indices.open(self.es_index_name)

        self.eval_dataset = self.eval_dataset.map(self.searcher, fn_kwargs=self.fn_kwargs, batched=True, **self.map_kwargs)
        report = ranx.compare(
            self.searcher.qrels,
            runs=self.searcher.runs.values(),
            **self.searcher.metrics_kwargs
        )
        return report


def get_objective(objective_type, train_dataset, **objective_kwargs):
    """
    Instantiates Objective.
    
    Parameters
    ----------
    objective_type: str
        'bm25' to use BM25Objective
        BM25Objective recomputes relevant results every time
    train_dataset: Dataset
    **objective_kwargs:
        passed to Objective
        
    Returns
    -------
    objective: Objective
    default_study_kwargs: dict
        Default sampler (GridSampler) and direction ('maximize'), 
        passed to optuna.create_study if not overwritten.
    """
    if objective_type == 'bm25':
        objective = BM25Objective(train_dataset, do_cache_relevant=False, **objective_kwargs)
        hyp_hyp = objective.hyp_hyp
        search_space = dict(b=np.arange(*hyp_hyp['b']["bounds"], hyp_hyp['b']["step"]).tolist(),
                            k1=np.arange(*hyp_hyp['k1']["bounds"], hyp_hyp['k1']["step"]).tolist())
        default_study_kwargs = dict(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
    else:
        raise ValueError(f"Invalid objective type: {objective_type}")
    return objective, default_study_kwargs


def hyperparameter_search(study_name=None, storage=None, metric_save_path=None,
                          optimize_kwargs={}, study_kwargs={}, cleanup_cache_files=False, **objective_kwargs):
    """
    Main function that loads data, caches relevant results according to objective.do_cache_relevant,
    runs the optimization and saves the results.
    
    Parameters
    ----------
    study_name, storage: 
        see optuna.create_study
    metric_save_path: str, optional
        Path to the directory where to save the results qrels, runs and metrics of eval_dataset.
        Defaults not to save.
    optimize_kwargs: dict, optional
        Passed to Study.optimize
    study_kwargs: dict, optional
        Passed to optuna.create_study, default is defined in ``get_objective``
    cleanup_cache_files: bool, optional
        Clean up all cache files in the dataset cache directory, 
        excepted the currently used one, see Dataset.cleanup_cache_files()
    **objective_kwargs:
        Passed to Objective
    """
    objective, default_study_kwargs = get_objective(cleanup_cache_files=cleanup_cache_files, **objective_kwargs)
    default_study_kwargs.update(study_kwargs)
    sampler = default_study_kwargs.get('sampler')
    # get sampler by name
    if isinstance(sampler, str):
        default_study_kwargs['sampler'] = getattr(optuna.samplers, sampler)(**default_study_kwargs.get('sampler_kwargs', {}))
    if storage is None and study_name is not None:
        storage = f"sqlite:///{study_name}.db"
    study = optuna.create_study(storage=storage, study_name=study_name, load_if_exists=True, **default_study_kwargs)
    print(f"Using a sampler of type {type(study.sampler)}")
    if objective.do_cache_relevant and objective.dataset is not None:
        objective.cache_relevant_dataset()
    # actual optimisation
    if objective.dataset is not None:
        study.optimize(objective, **optimize_kwargs)
    print(f"Best value: {study.best_value} ({objective.metric_for_best_model})")
    print(f"Best hyperparameters: {study.best_params}")

    # apply hyperparameters on test set
    if eval_dataset is not None:
        if objective.do_cache_relevant and objective.dataset is not None:
            objective.searcher.reference_kb = objective.keep_reference_kb
        report = objective.evaluate(study.best_params)
        print(report)

        if metric_save_path is not None:
            metric_save_path = Path(metric_save_path)
            metric_save_path.mkdir(exist_ok=True)
            # N. B. qrels and runs are overwritten in Searcher every time there's a call to add_multi
            objective.searcher.qrels.save(metric_save_path / "qrels.json")
            report.save(metric_save_path / "metrics.json")
            with open(metric_save_path / "metrics.tex", 'wt') as file:
                file.write(report.to_latex())
            for index_name, run in objective.searcher.runs.items():
                run.save(metric_save_path / f"{index_name}.json")

    return objective.eval_dataset

if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    cleanup_cache_files = args['--cleanup_cache_files']
    config_path = args['<config>']
    with open(config_path, 'r') as file:
        config = json.load(file)
    format_kwargs = config.pop('format', {})

    if args['--train'] is not None:
        dataset = load_from_disk(args['--train'])
        dataset.set_format(**format_kwargs)
    else:
        dataset = None

    k = int(args['--k'])

    eval_dataset_path = args['--test']
    if eval_dataset_path is not None:
        eval_dataset = load_from_disk(eval_dataset_path)
        eval_dataset.set_format(**format_kwargs)
    else:
        eval_dataset = None
    eval_dataset = hyperparameter_search(objective_type=args['<type>'], train_dataset=dataset, k=k,
                                         metric_save_path=args['--metrics'], eval_dataset=eval_dataset, 
                                         cleanup_cache_files=cleanup_cache_files, **config)
    if eval_dataset is not None:
        eval_dataset.save_to_disk(eval_dataset_path)