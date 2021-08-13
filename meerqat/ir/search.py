"""Both dense and sparse information retrieval is done via HF-Datasets, using FAISS and ElasticSearch, respectively

Usage:
search.py <dataset> <config> [--k=<k> --disable_caching --save_irrelevant --metrics=<path>]
search.py hp <dataset> <config> [--k=<k> --disable_caching --metrics=<path>]

Options:
--k=<k>                 Hyperparameter to search for the k nearest neighbors [default: 100].
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--save_irrelevant       Save irrelevant results from the search in the dataset
--metrics=<path>        Path to save the results in JSON format
"""
import warnings

from docopt import docopt
import json
from collections import Counter
import time
from copy import deepcopy

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
        # fusion = es + alpha * faiss
        for index, score in faiss_dict.items():
            es_dict.setdefault(index, 0.)
            es_dict[index] += alpha * score
        # sort in desc. order and keep top-k
        scores, indices = dict2scores(es_dict, k=k)
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


def fuse_and_compute_metrics(batch, kbs, metrics, metrics_kwargs={}, reference_kb=None, reference_key='passage', k=100, fusion_method='linear', **fusion_kwargs):
    scores_batch, indices_batch = fuse(batch, kbs, k=k, method=fusion_method, **fusion_kwargs)
    batch['scores'], batch['indices'] = scores_batch, indices_batch

    # are the retrieved documents relevant ?
    if reference_kb is not None:
        relevant_batch = find_relevant_batch(indices_batch, batch['output'], reference_kb,
                                             relevant_batch=deepcopy(batch['provenance_index']), reference_key=reference_key)
    else:
        relevant_batch = batch['provenance_index']

    # compute metrics
    compute_metrics(metrics["fusion"],
                    retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                    K=k, scores_batch=scores_batch, **metrics_kwargs)
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
                                                 relevant_batch=deepcopy(batch['provenance_index']), reference_key=reference_key)
        else:
            relevant_batch = batch['provenance_index']

        # compute metrics
        compute_metrics(metrics[index_name],
                        retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                        K=k, scores_batch=scores_batch, **metrics_kwargs)
        
        if save_irrelevant:
            irrelevant_batch = get_irrelevant_batch(retrieved_batch=indices_batch, relevant_batch=relevant_batch)
            batch[f'{index_name}_irrelevant_indices'] = irrelevant_batch

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


def index_faiss_kb(path, column, index_name=None, load=False, save_path=None, **kwargs):
    """
    Loads KB from path then either:
    - loads it (if load)
    - default: applies Dataset.add_faiss_index method (identical parameters), and save it if save_path is not None
    """
    if index_name is None:
        index_name = column
    kb = load_from_disk(path)
    if load:
        kb.load_faiss_index(**kwargs)
    else:
        kb.add_faiss_index(column=column, index_name=index_name, **kwargs)
        # save FAISS index (so it can be loaded later)
        if save_path is not None:
            kb.save_faiss_index(index_name, save_path)
    return kb, index_name


def dataset_search(dataset, k=100, save_irrelevant=False, metric_save_path=None,
                   kb_kwargs=[], map_kwargs={}, fusion_kwargs={}, metrics_kwargs={}):
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
            string_factory = index_kwargs.get('string_factory')
            if string_factory is not None and 'L2norm' in string_factory:
                kb_kwarg['L2norm'] = True
            kb, index_name = index_faiss_kb(**index_kwargs)

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

    # search expects a batch as input
    fn_kwargs = dict(kbs=kbs, reference_kb=reference_kb, k=k, save_irrelevant=save_irrelevant, metrics=metrics, metrics_kwargs=metrics_kwargs, **fusion_kwargs)

    # HACK: sleep until elasticsearch is good to go
    time.sleep(60)

    dataset = dataset.map(search, fn_kwargs=fn_kwargs, batched=True, **map_kwargs)

    reduce_metrics(metrics, K=k)
    print(stringify_metrics(metrics, tablefmt='latex', floatfmt=".2f"))
    if metric_save_path is not None:
        with open(metric_save_path, 'w') as file:
            json.dump(metrics, file)

    return dataset


class FusionObjective:
    """Callable objective compatible with optuna. Holds data necessary to run fuse_and_compute_metrics"""
    def __init__(self, dataset, k=100, kbs=None, fusion_method='linear', hyp_hyp=None, metric_for_best_model=None,
                 fn_kwargs={}, fusion_kwargs={}, map_kwargs={}):
        self.dataset = dataset
        self.k = k
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
        if metric_for_best_model is None:
            self.metric_for_best_model = f"MRR@{self.k}"
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


def hyperparameter_search(dataset, k=100, metric_save_path=None, optimize_kwargs={}, study_kwargs={}, **objective_kwargs):
    objective = FusionObjective(dataset, k=k, **objective_kwargs)
    if objective.fusion_method == 'linear':
        alpha_hyp = objective.hyp_hyp[objective.fusion_method]['alpha']
        search_space = dict(alpha=np.arange(*alpha_hyp["bounds"], alpha_hyp["step"]))
        default_study_kwargs = dict(direction='maximize', sampler=optuna.samplers.GridSampler(search_space))
    else:
        default_study_kwargs = {}
    default_study_kwargs.update(study_kwargs)
    study = optuna.create_study(**default_study_kwargs)
    # actual optimisation
    study.optimize(objective, **optimize_kwargs)
    print(f"Best value: {study.best_value} (should match {objective.metric_for_best_model})")
    print(f"Best hyperparameters: {study.best_params}")
    best_trial = study.best_trial
    metrics = best_trial.user_attrs.get('metrics')
    if metrics is not None:
        print(stringify_metrics(metrics, tablefmt='latex', floatfmt=".2f"))
        if metric_save_path is not None:
            with open(metric_save_path, 'w') as file:
                json.dump(metrics, file)


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
        hyperparameter_search(dataset, k, metric_save_path=args['--metrics'], **config)
    else:
        dataset = dataset_search(dataset, k,
                                 save_irrelevant=args['--save_irrelevant'],
                                 metric_save_path=args['--metrics'],
                                 **config)

        dataset.save_to_disk(dataset_path)
