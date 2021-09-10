"""Both dense and sparse information retrieval is done via HF-Datasets, using FAISS and ElasticSearch, respectively

Usage:
search.py <dataset> <config> [--k=<k> --disable_caching --save_irrelevant --metrics=<path>]
search.py hp <type> <dataset> <config> [--k=<k> --disable_caching --metrics=<path> --test=<dataset>]

Options:
--k=<k>                 Hyperparameter to search for the k nearest neighbors [default: 100].
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--save_irrelevant       Save 1. irrelevant results from the search, 2. the union of relevant search and provenance_indices
--metrics=<path>        Path to save the results in JSON format
--test=<dataset>        Name of the test dataset
"""
import warnings

from docopt import docopt
import json
from collections import Counter
import time
from copy import deepcopy
import re

import numpy as np
from datasets import load_from_disk, set_caching_enabled

import optuna

from meerqat.ir.metrics import compute_metrics, reduce_metrics, stringify_metrics, find_relevant_batch, get_irrelevant_batch
from meerqat.data.utils import json_integer_keys


def scores2dict(scores_batch, indices_batch):
    scores_dicts = []
    for scores, indices in zip(scores_batch, indices_batch):
        scores_dicts.append(dict(zip(indices, scores)))
    return scores_dicts


def dict2scores(scores_dict, k=100):
    """sort in desc. order and keep top-k"""
    indices = sorted(scores_dict, key=scores_dict.get, reverse=True)[:k]
    scores = [scores_dict[index] for index in indices]
    return scores, indices


def dict_batch2scores(scores_dicts, k=100):
    scores_batch, indices_batch = [], []
    for scores_dict in scores_dicts:
        scores, indices = dict2scores(scores_dict, k=k)
        scores_batch.append(scores)
        indices_batch.append(indices)
    return scores_batch, indices_batch


def split_es_and_faiss_kbs(kbs):
    es_kbs, faiss_kbs = [], []
    for kb in kbs:
        if kb['es']:
            es_kbs.append(kb)
        else:
            faiss_kbs.append(kb)
    return es_kbs, faiss_kbs


def set_interpolation_weights(kbs):
    """See interpolation_fusion"""
    total, not_set = 0, 0
    for kb in kbs:
        weight = kb.get('weight')
        if weight is None:
            not_set += 1
        else:
            total += weight
    # all weights are already set
    if not_set == 0:
        return kbs
    assert total <= 1, f"All weights should sum to 1 but you've not set {not_set} weights and the other sum to {total}"

    # set uniform weight such that they all sum to 1
    uniform_weight = (1-total)/not_set
    for kb in kbs:
        kb.setdefault('weight', uniform_weight)
    return kbs


def interpolation_fusion(batch, kbs, k=100):
    """
    Simple weighted sum, e.g. : fusion = w_1*score_1 + w_2*score_2 + w_3*score_3

    If the weight are partially provided or not provided at all they default to a uniform weight such that they all sum to 1.
    """
    batch_size = len(next(iter(batch.values())))
    # init scores
    # N. B. [{}]*n creates n pointers to the SAME dict
    scores_dicts = [{} for _ in range(batch_size)]
    kbs = set_interpolation_weights(kbs)
    for kb in kbs:
        weight = kb['weight']
        index_name = kb['index_name']

        kb_scores_dicts = scores2dict(batch[f'{index_name}_scores'], batch[f'{index_name}_indices'])
        for scores_dict, kb_scores_dict in zip(scores_dicts, kb_scores_dicts):
            for index, score in kb_scores_dict.items():
                scores_dict.setdefault(index, 0.)
                scores_dict[index] += weight * score

    scores_batch, indices_batch = dict_batch2scores(scores_dicts, k=k)
    return scores_batch, indices_batch


def linear_fusion(batch, kbs, k=100, alpha=1.1):
    """
    fuses sparse and dense search following Karpukhin et. al : fusion = es + alpha * faiss

    If there are multiple ES KBs or FAISS KBs, they are first fused separately using a simple interpolation
    """

    # If there are multiple ES KBs or FAISS KBs, they are first fused separately using a simple interpolation
    es_kbs, faiss_kbs = split_es_and_faiss_kbs(kbs)

    if len(es_kbs) > 1:
        es_scores, es_indices = fuse(batch, es_kbs, k=k, method='interpolation')
    else:
        index_name = es_kbs[0]['index_name']
        es_scores, es_indices = batch[f'{index_name}_scores'], batch[f'{index_name}_indices']
    if len(faiss_kbs) > 1:
        faiss_scores, faiss_indices = fuse(batch, faiss_kbs, k=k, method='interpolation')
    else:
        index_name = faiss_kbs[0]['index_name']
        faiss_scores, faiss_indices = batch[f'{index_name}_scores'], batch[f'{index_name}_indices']

    # once there is only one ES and one FAISS retrieval left, proceed to the linear fusion
    scores_batch, indices_batch = [], []
    es_dicts = scores2dict(es_scores, es_indices)
    faiss_dicts = scores2dict(faiss_scores, faiss_indices)

    for es_dict, faiss_dict in zip(es_dicts, faiss_dicts):
        # FIXME this is not consistent with interpolation_fusion
        # follow Ma et al. (2021, arXiv:2104.05740) by using minimal score when the document was retrieved only by one system
        min_es_score = min(es_dict.values())
        min_faiss_score = min(faiss_dict.values())
        fusion_dict = {}
        for index in set(es_dict.keys()) | set(faiss_dict.keys()):
            es_score = es_dict.get(index, min_es_score)
            faiss_score = faiss_dict.get(index, min_faiss_score)
            fusion_dict[index] = es_score + alpha * faiss_score
        # sort in desc. order and keep top-k
        scores, indices = dict2scores(fusion_dict, k=k)
        scores_batch.append(scores)
        indices_batch.append(indices)

    return scores_batch, indices_batch


def fuse(batch, kbs, k=100, method='linear', **kwargs):
    """Should return a (scores, indices) tuples the same way as Dataset.search_batch"""

    # easy to fuse when there is only one input
    if len(kbs) == 1:
        index_name = kbs[0]['index_name']
        return batch[f'{index_name}_scores'], batch[f'{index_name}_indices']

    fusions = dict(linear=linear_fusion, interpolation=interpolation_fusion)

    return fusions[method](batch, kbs, k=k, **kwargs)


def map_indices(scores_batch, indices_batch, mapping, k=None):
    """
    Also takes scores as argument to align scores and indices in case of 1-many mapping

    If k is not None, keep only the top-k (might have exceeded in case of 1-many mapping)
    """
    new_scores_batch, new_indices_batch = [], []
    for scores, indices in zip(scores_batch, indices_batch):
        new_scores, new_indices = [], []
        for score, index in zip(scores, indices):
            # extend because it can be a 1-many mapping (e.g. document/image to passage)
            new_indices.extend(mapping[index])
            new_scores.extend([score]*len(mapping[index]))
            if k is not None and len(new_indices) >= k:
                break
        new_scores_batch.append(new_scores)
        new_indices_batch.append(new_indices)
    return new_scores_batch, new_indices_batch


def norm_mean_std(scores_batch, mean, std):
    return [(np.array(scores)-mean)/std for scores in scores_batch]


def normalize(scores_batch, method, **kwargs):
    methods = {
        "normalize": norm_mean_std
    }
    return methods[method](scores_batch, **kwargs)


def L2norm(queries):
    """Normalize each query to have a unit-norm. Expects a batch of vectors of the same dimension"""
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    return queries/norms


def search_batch(kb, index_name, queries, k=100):
    if not kb['es']:
        queries = np.array(queries, dtype=np.float32)
        if kb.get('L2norm', False):
            queries = L2norm(queries)
    return kb['kb'].search_batch(index_name, queries, k=k)


def search_batch_if_not_None(kb, index_name, queries, k=100):
    # 1. filter out queries that are None
    scores_batch, indices_batch = [], []
    not_None_queries, not_None_queries_indices = [], []
    for i, query in enumerate(queries):
        # default to empty (i.e. no result with None query)
        # will be overwritten for not_None_queries
        scores_batch.append([])
        indices_batch.append([])
        if query is not None:
            not_None_queries.append(query)
            not_None_queries_indices.append(i)
    if not not_None_queries:
        return scores_batch, indices_batch
            
    # 2. search as usual for queries that are not None
    not_None_queries = np.array(not_None_queries, dtype=np.float32)
    not_None_scores_batch, not_None_indices_batch = search_batch(kb, index_name, not_None_queries, k=k)

    # 3. return the results in a list of list with proper indices
    for j, i in enumerate(not_None_queries_indices):
        scores_batch[i] = not_None_scores_batch[j]
        indices_batch[i] = not_None_indices_batch[j]
    return scores_batch, indices_batch


def fuse_and_compute_metrics(batch, kbs, metrics, metrics_kwargs={}, reference_kb=None, reference_key='passage', k=100, fusion_method='linear', save_irrelevant=False, **fusion_kwargs):
    scores_batch, indices_batch = fuse(batch, kbs, k=k, method=fusion_method, **fusion_kwargs)
    batch['search_scores'], batch['search_indices'] = scores_batch, indices_batch

    # are the retrieved documents relevant ?
    if reference_kb is not None:
        relevant_batch = find_relevant_batch(indices_batch, batch['output'], reference_kb,
                                             relevant_batch=deepcopy(batch['provenance_indices']), reference_key=reference_key)
    else:
        relevant_batch = batch['provenance_indices']

    # compute metrics
    compute_metrics(metrics["fusion"],
                    retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                    K=k, scores_batch=scores_batch, **metrics_kwargs)

    if save_irrelevant:
        irrelevant_batch = get_irrelevant_batch(retrieved_batch=indices_batch, relevant_batch=relevant_batch)
        batch[f'search_irrelevant_indices'] = irrelevant_batch
        batch[f'search_provenance_indices'] = relevant_batch
    return batch


def search(batch, kbs, reference_kb=None, k=100, metrics={}, metrics_kwargs={}, reference_key='passage', fusion_method='linear', save_irrelevant=False, **fusion_kwargs):
    # search with the KBs
    for kb in kbs:
        index_name = kb['index_name']
        queries = batch[kb['key']]
        # N. B. cannot use `None in queries` because 
        # "The truth value of an array with more than one element is ambiguous."
        if any(query is None for query in queries):
            scores_batch, indices_batch = search_batch_if_not_None(kb, index_name, queries, k=k)
        else:
            scores_batch, indices_batch = search_batch(kb, index_name, queries, k=k)

        # indices might need to be mapped so that all KBs refer to the same semantic index
        index_mapping = kb.get('index_mapping')
        if index_mapping:
            scores_batch, indices_batch = map_indices(scores_batch, indices_batch, index_mapping, k=k)

        # eventually normalize the scores before fusing
        normalization = kb.get('normalization')
        if normalization is not None:
            scores_batch = normalize(scores_batch, **normalization)

        # store result in the dataset
        batch[f'{index_name}_scores'] = scores_batch
        batch[f'{index_name}_indices'] = indices_batch

        # are the retrieved documents relevant ?
        if reference_kb is not None:
            relevant_batch = find_relevant_batch(indices_batch, batch['output'], reference_kb,
                                                 relevant_batch=deepcopy(batch['provenance_indices']), reference_key=reference_key)
        else:
            relevant_batch = batch['provenance_indices']

        # compute metrics
        compute_metrics(metrics[index_name],
                        retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                        K=k, scores_batch=scores_batch, **metrics_kwargs)
        
        if save_irrelevant:
            irrelevant_batch = get_irrelevant_batch(retrieved_batch=indices_batch, relevant_batch=relevant_batch)
            batch[f'{index_name}_irrelevant_indices'] = irrelevant_batch
            batch[f'{index_name}_provenance_indices'] = relevant_batch

    # fuse the results of the searches
    if len(kbs) > 1:
        fuse_and_compute_metrics(batch, kbs, metrics,
                                 metrics_kwargs=metrics_kwargs, reference_key=reference_key,
                                 reference_kb=reference_kb, k=k, fusion_method=fusion_method, **fusion_kwargs)
    return batch


def index_es_kb(path, column, index_name=None, load=False, **kwargs):
    """
    Loads KB from path then either:
    - loads it (if load)
    - default: applies Dataset.add_elasticsearch_index method (identical parameters),
      the index is then saved by the ElasticSearch server
    """
    if index_name is None:
        index_name = column
    kb = load_from_disk(path)
    if load:
        kb.load_elasticsearch_index(index_name=index_name, **kwargs)
    else:
        kb.add_elasticsearch_index(column=column, index_name=index_name, **kwargs)
    return kb, index_name


def index_faiss_kb(path, column, index_name=None, load=False, save_path=None, string_factory=None, device=None, **kwargs):
    """
    Loads KB from path then either:
    - loads it (if load)
    - default: applies Dataset.add_faiss_index method (identical parameters), and save it if save_path is not None
    """
    if index_name is None:
        index_name = column
    kb = load_from_disk(path)
    if string_factory is not None and 'L2norm' in string_factory:
        do_L2norm = True
    else:
        do_L2norm = False
    if load:
        kb.load_faiss_index(**kwargs)
    else:
        # HACK: fix L2-normalisation on GPU https://github.com/facebookresearch/faiss/issues/2010
        if do_L2norm and device is not None:
            # normalize the vectors
            kb = kb.map(lambda batch: {column: L2norm(batch[column])}, batched=True)
            # remove "L2norm" from string_factory
            string_factory = re.sub(r"(,L2norm|L2norm[,]?)", "", string_factory)
            if not string_factory:
                string_factory = None
        kb.add_faiss_index(column=column, index_name=index_name, string_factory=string_factory, device=device, **kwargs)
        # save FAISS index (so it can be loaded later)
        if save_path is not None:
            kb.save_faiss_index(index_name, save_path)
    return kb, index_name, do_L2norm


def load_kbs(kb_kwargs):
    kbs = []
    index_names = set()
    metrics = {}
    # find the kb with reference indices
    reference_kb = None
    # load KB, index it and load index-mapping, if relevant
    for kb_kwarg in kb_kwargs:
        # load and index KB
        es = kb_kwarg.pop('es', False)
        index_kwargs = kb_kwarg.pop('index_kwargs', {})
        if es:
            kb, index_name = index_es_kb(**index_kwargs)
        else:
            kb, index_name, do_L2norm = index_faiss_kb(**index_kwargs)
            kb_kwarg['L2norm'] = do_L2norm

        # index mapping are used so that multiple KBs are aligned to the same semantic indices
        # e.g. text is processed at the passage level but images are processed at the document/entity level
        index_mapping = kb_kwarg.get('index_mapping')
        if index_mapping is not None:
            with open(index_mapping, 'r') as file:
                # convert all keys to int (JSON unfortunately does not support integer keys)
                kb_kwarg['index_mapping'] = json.load(file, object_hook=json_integer_keys)

        if reference_kb is None and kb_kwarg.get("reference", False):
            reference_kb = kb

        kbs.append(dict(kb=kb, index_name=index_name, es=es, **kb_kwarg))
        assert index_name not in index_names, "All KBs should have unique index names"
        assert index_name not in {'search', 'fusion'}, "'search', 'fusion' are reserved names"
        index_names.add(index_name)

        # initialize the metrics for this KB
        metrics[index_name] = Counter()

    assert len(index_names) >= 1, 'Expected at least one KB'
    if len(index_names) > 1:
        metrics["fusion"] = Counter()

    if reference_kb is None:
        warnings.warn("Didn't find a reference KB "
                      "-> will not be able to extend the annotation coverage so results should be interpreted carefully.\n"
                      "Did you forget to add a 'reference' flag to your config file?")

    return dict(kbs=kbs, reference_kb=reference_kb, metrics=metrics)


def dataset_search(dataset, k=100, save_irrelevant=False, metric_save_path=None,
                   kb_kwargs=[], map_kwargs={}, fusion_kwargs={}, metrics_kwargs={}):
    fn_kwargs = dict(k=k, save_irrelevant=save_irrelevant, metrics_kwargs=metrics_kwargs, **fusion_kwargs)
    kbs = load_kbs(kb_kwargs)
    fn_kwargs.update(kbs)

    # HACK: sleep until elasticsearch is good to go
    time.sleep(60)

    # search expects a batch as input
    dataset = dataset.map(search, fn_kwargs=fn_kwargs, batched=True, **map_kwargs)

    metrics = fn_kwargs['metrics']
    reduce_metrics(metrics, K=k)
    print(stringify_metrics(metrics, tablefmt='latex', floatfmt=".2f"))
    if metric_save_path is not None:
        with open(metric_save_path, 'w') as file:
            json.dump(metrics, file)

    return dataset


class Objective:
    """Callable objective compatible with optuna."""
    def __init__(self, dataset, k=100, metric_for_best_model=None, eval_dataset=None):
        self.dataset = dataset
        self.k = k
        if metric_for_best_model is None:
            self.metric_for_best_model = f"MRR@{self.k}"
        else:
            self.metric_for_best_model = metric_for_best_model
        self.eval_dataset = eval_dataset

    def __call__(self, trial):
        pass

    def evaluate(self, best_params):
        """
        Should evaluate self.eval_dataset with best_params

        Parameters
        ----------
        best_params: dict

        Returns
        -------
        metrics: dict
        """
        pass

    def prefix_eval(self, eval_metrics):
        for k in list(eval_metrics.keys()):
            eval_metrics['eval_'+k] = eval_metrics.pop(k)
        return eval_metrics


class FusionObjective(Objective):
    def __init__(self, kbs=None, fusion_method='linear', hyp_hyp=None,
                 fn_kwargs={}, fusion_kwargs={}, map_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_method = fusion_method
        self.fusion_kwargs = fusion_kwargs
        self.map_kwargs = map_kwargs

        # default parameters
        if hyp_hyp is None:
            self.hyp_hyp = {
                'linear': {
                    "alpha": {
                        "bounds": (0, 2),
                        "step": 0.1
                    }
                }
            }
        else:
            self.hyp_hyp = hyp_hyp

        reference_kb = None
        for kb in kbs:
            if kb.get("reference", False):
                reference_kb = load_from_disk(kb.pop('path'))
                break
        if reference_kb is None:
            warnings.warn("Didn't find a reference KB "
                          "-> will not be able to extend the annotation coverage so results should be interpreted carefully.\n"
                          "Did you forget to add a 'reference' flag to your config file?")

        fn_kwargs.update(dict(kbs=kbs, reference_kb=reference_kb, k=self.k, fusion_method=self.fusion_method))
        self.fn_kwargs = fn_kwargs

    def __call__(self, trial):
        fusion_kwargs = self.fusion_kwargs
        fn_kwargs = self.fn_kwargs
        if self.fusion_method == 'linear':
            alpha = trial.suggest_float("alpha", *self.hyp_hyp[self.fusion_method]["alpha"]["bounds"])
            fusion_kwargs['alpha'] = alpha
        else:
            raise NotImplementedError()
        fn_kwargs.update(fusion_kwargs)

        metrics = {"fusion": Counter()}
        fn_kwargs['metrics'] = metrics
        self.dataset.map(fuse_and_compute_metrics, fn_kwargs=fn_kwargs, batched=True, **self.map_kwargs)
        reduce_metrics(metrics, K=self.k)
        trial.set_user_attr('metrics', metrics)
        return metrics['fusion'][self.metric_for_best_model]

    def evaluate(self, best_params):
        fn_kwargs = self.fn_kwargs
        fn_kwargs.update(best_params)
        eval_metrics = {"fusion": Counter()}
        fn_kwargs['metrics'] = eval_metrics
        self.eval_dataset = self.eval_dataset.map(fuse_and_compute_metrics, fn_kwargs=fn_kwargs, batched=True, **self.map_kwargs)
        reduce_metrics(eval_metrics, K=k)
        return self.prefix_eval(eval_metrics)


class BM25Objective(Objective):
    def __init__(self, *args, kb_kwargs=None, hyp_hyp=None, settings=None,
                 fn_kwargs={}, fusion_kwargs={}, map_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_kwargs = fusion_kwargs
        self.map_kwargs = map_kwargs

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

        fn_kwargs['k'] = self.k
        kbs = load_kbs(kb_kwargs)
        es_kbs, _ = split_es_and_faiss_kbs(kbs['kbs'])
        if len(es_kbs) != 1:
            raise ValueError(f"Expected exactly 1 ES KB, got {len(es_kbs)}")
        self.es_kb = es_kbs[0]
        self.index_name = self.es_kb['index_name']
        es_index = self.es_kb['kb']._indexes[self.index_name]
        self.es_client = es_index.es_client
        self.es_index_name = es_index.es_index_name
        fn_kwargs.update(kbs)
        self.fn_kwargs = fn_kwargs

    def __call__(self, trial):
        fn_kwargs = self.fn_kwargs
        kbs = self.fn_kwargs['kbs']
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

        metrics = {kb['index_name']: Counter() for kb in kbs}
        if len(kbs) > 1:
            metrics["fusion"] = Counter()
        fn_kwargs['metrics'] = metrics

        self.dataset.map(search, fn_kwargs=fn_kwargs, batched=True, **self.map_kwargs)
        reduce_metrics(metrics, K=self.k)

        trial.set_user_attr('metrics', metrics)
        metric = metrics.get('fusion', metrics[self.index_name])
        return metric[self.metric_for_best_model]

    def evaluate(self, best_params):
        fn_kwargs = self.fn_kwargs
        kbs = self.fn_kwargs['kbs']
        settings = self.settings

        for parameters in settings['similarity'].values():
            parameters.update(best_params)
        # close index, update its settings then open it
        self.es_client.indices.close(self.es_index_name)
        self.es_client.indices.put_settings(settings, self.es_index_name)
        self.es_client.indices.open(self.es_index_name)

        metrics = {kb['index_name']: Counter() for kb in kbs}
        if len(kbs) > 1:
            metrics["fusion"] = Counter()
        fn_kwargs['metrics'] = metrics

        self.eval_dataset = self.eval_dataset.map(search, fn_kwargs=fn_kwargs, batched=True, **self.map_kwargs)
        reduce_metrics(metrics, K=self.k)

        return self.prefix_eval(metrics)


def get_objective(objective_type, train_dataset, k=100, **objective_kwargs):
    if objective_type == 'fusion':
        objective = FusionObjective(train_dataset, k=k, **objective_kwargs)
        if objective.fusion_method == 'linear':
            alpha_hyp = objective.hyp_hyp[objective.fusion_method]['alpha']
            search_space = dict(alpha=np.arange(*alpha_hyp["bounds"], alpha_hyp["step"]).tolist())
            default_study_kwargs = dict(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
        else:
            default_study_kwargs = {}
    elif objective_type == 'bm25':
        objective = BM25Objective(train_dataset, k=k, **objective_kwargs)
        hyp_hyp = objective.hyp_hyp
        search_space = dict(b=np.arange(*hyp_hyp['b']["bounds"], hyp_hyp['b']["step"]).tolist(),
                            k1=np.arange(*hyp_hyp['k1']["bounds"], hyp_hyp['k1']["step"]).tolist())
        default_study_kwargs = dict(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
    else:
        raise ValueError(f"Invalid objective type: {objective_type}")
    return objective, default_study_kwargs


def hyperparameter_search(metric_save_path=None,
                          optimize_kwargs={}, study_kwargs={}, **objective_kwargs):
    objective, default_study_kwargs = get_objective(**objective_kwargs)
    default_study_kwargs.update(study_kwargs)
    study = optuna.create_study(**default_study_kwargs)
    # actual optimisation
    study.optimize(objective, **optimize_kwargs)
    print(f"Best value: {study.best_value} (should match {objective.metric_for_best_model})")
    print(f"Best hyperparameters: {study.best_params}")
    best_trial = study.best_trial
    metrics = best_trial.user_attrs.get('metrics')

    # apply hyperparameters on test set
    if eval_dataset is not None:
        eval_metrics = objective.evaluate(study.best_params)
        metrics.update(eval_metrics)

    if metrics is not None:
        print(stringify_metrics(metrics, tablefmt='latex', floatfmt=".2f"))
        if metric_save_path is not None:
            with open(metric_save_path, 'w') as file:
                json.dump(metrics, file)

    return objective.eval_dataset


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_path = args['<dataset>']
    dataset = load_from_disk(dataset_path)
    set_caching_enabled(not args['--disable_caching'])
    config_path = args['<config>']
    with open(config_path, 'r') as file:
        config = json.load(file)
    format_kwargs = config.pop('format', {})
    dataset.set_format(**format_kwargs)

    k = int(args['--k'])

    if args['hp']:
        # TODO optimize BM25
        eval_dataset_path = args['--test']
        if eval_dataset_path:
            eval_dataset = load_from_disk(eval_dataset_path)
        else:
            eval_dataset = None
        eval_dataset = hyperparameter_search(objective_type=args['<type>'], train_dataset=dataset, k=k,
                                             metric_save_path=args['--metrics'], eval_dataset=eval_dataset, **config)
        if eval_dataset is not None:
            eval_dataset.save_to_disk(eval_dataset_path)
    else:
        dataset = dataset_search(dataset, k,
                                 save_irrelevant=args['--save_irrelevant'],
                                 metric_save_path=args['--metrics'],
                                 **config)

        dataset.save_to_disk(dataset_path)
