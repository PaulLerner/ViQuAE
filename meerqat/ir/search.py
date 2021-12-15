"""Both dense and sparse information retrieval is done via HF-Datasets, using FAISS and ElasticSearch, respectively

Usage:
search.py <dataset> <config> [--k=<k> --disable_caching --metrics=<path>]

Options:
--k=<k>                 Hyperparameter to search for the k nearest neighbors [default: 100].
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--metrics=<path>        Path to save the results in JSON format
"""
import warnings

from docopt import docopt
import json
from collections import Counter
import time
from copy import deepcopy
import re
import enum
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch
from datasets import load_from_disk, set_caching_enabled
from datasets.search import ElasticSearchIndex, FaissIndex

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


class IndexKind(enum.Enum):
    """Will be used later on for data-specific fusion"""
    TEXT = enum.auto()
    FACE = enum.auto()
    # short for "full-image", as opposed to "FACE"
    IMAGE = enum.auto()


class Index:
    """
    N. B. difficult to create a hierarchy like FaissIndex and ESIndex since public methods, 
    such as search_batch, are defined in Dataset and take as input the index name.
    """
    def __init__(self, key, kind_str=None, es=False, do_L2norm=False, normalization=None, interpolation_weight=None):
        self.key = key
        if kind_str is None:
            self.kind = None
        else:
            self.kind = IndexKind[kind_str]
        self.es = es
        self.do_L2norm = do_L2norm
        self.normalization = normalization
        self.interpolation_weight = interpolation_weight


class KnowledgeBase:
    """A KB can be indexed by several indexes."""
    def __init__(self, kb_path=None, index_mapping_path=None, index_kwargs={}, es_client=None, load_dataset=True):
        if load_dataset:
            self.dataset = load_from_disk(kb_path)
        # This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp).
        else:
            self.dataset = None
        self.es_client = es_client

        # N. B. this dict[Index] holds extra informations about the indexes. 
        # to access actual HF indexes, use self.dataset._indexes
        self.indexes = {}
        if index_mapping_path is None:
            self.index_mapping = None
        else:
            with open(index_mapping_path, 'rt') as file:
                # convert all keys to int (JSON unfortunately does not support integer keys)
                self.index_mapping = json.load(file, object_hook=json_integer_keys)

        for index_name, index_kwarg in index_kwargs.items():
            self.add_or_load_index(index_name=index_name, **index_kwarg)


    def search_batch(self, index_name, queries, k=100):
        """Pre-process queries according to index before computing self.dataset.search_batch"""
        index = self.indexes[index_name]
        # N. B. should be equivalent to isinstance(self.dataset._indexes[index_name], FaissIndex)
        if not index.es:
            queries = np.array(queries, dtype=np.float32)
            if index.do_L2norm:
                queries = L2norm(queries)
        return self.dataset.search_batch(index_name, queries, k=k)

    def search_batch_if_not_None(self, index_name, queries, k=100):
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
        not_None_scores_batch, not_None_indices_batch = self.search_batch(index_name, not_None_queries, k=k)

        # 3. return the results in a list of list with proper indices
        for j, i in enumerate(not_None_queries_indices):
            scores_batch[i] = not_None_scores_batch[j]
            indices_batch[i] = not_None_indices_batch[j]
        return scores_batch, indices_batch

    def map_indices(self, scores_batch, indices_batch, k=None):
        """
        Maps indices using self.index_mapping if it is not None. 
        Else returns the input (scores_batch, indices_batch)

        Also takes scores as argument to align scores and indices in case of 1-many mapping

        If k is not None, keep only the top-k (might have exceeded in case of 1-many mapping)

        Beware 'index'/'indices' here refers to integers outputs from the search. 
        Nothing to do with the Index class or self.dataset._indexes
        """
        if self.index_mapping is None:
            return scores_batch, indices_batch
        new_scores_batch, new_indices_batch = [], []
        for scores, indices in zip(scores_batch, indices_batch):
            new_scores, new_indices = [], []
            for score, index in zip(scores, indices):
                # extend because it can be a 1-many mapping (e.g. document/image to passage)
                new_indices.extend(self.index_mapping[index])
                new_scores.extend([score]*len(self.index_mapping[index]))
                if k is not None and len(new_indices) >= k:
                    break
            new_scores_batch.append(new_scores)
            new_indices_batch.append(new_indices)
        return new_scores_batch, new_indices_batch

    def add_or_load_index(self, column=None, index_name=None, es=False, kind_str=None, key=None,
                          normalization=None, interpolation_weight=None, **index_kwarg):
        # do not actually add the index. 
        # This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp).
        if column is None:
            do_L2norm = False
        else:
            if index_name is None:
                index_name = column
            if es:
                self.add_or_load_elasticsearch_index(column, index_name=index_name, **index_kwarg)
                do_L2norm = False
            else:                
                do_L2norm = self.add_or_load_faiss_index(column, index_name=index_name, **index_kwarg)
        index = Index(key=key, kind_str=kind_str, es=es, do_L2norm=do_L2norm, normalization=normalization, interpolation_weight=interpolation_weight)
        self.indexes[index_name] = index

    def add_or_load_faiss_index(self, column, index_name=None, load=False, save_path=None, string_factory=None, device=None, **kwargs):
        if string_factory is not None and 'L2norm' in string_factory:
            do_L2norm = True
        else:
            do_L2norm = False
        if load:
            self.dataset.load_faiss_index(**kwargs)
        else:
            # HACK: fix L2-normalisation on GPU https://github.com/facebookresearch/faiss/issues/2010
            if do_L2norm and device is not None:
                # normalize the vectors
                self.dataset = self.dataset.map(lambda batch: {column: L2norm(batch[column])}, batched=True)
                # remove "L2norm" from string_factory
                string_factory = re.sub(r"(,L2norm|L2norm[,]?)", "", string_factory)
                if not string_factory:
                    string_factory = None
            self.dataset.add_faiss_index(column, index_name=index_name, string_factory=string_factory, device=device, **kwargs)
            # save FAISS index (so it can be loaded later)
            if save_path is not None:
                self.dataset.save_faiss_index(index_name, save_path)
        return do_L2norm

    def add_or_load_elasticsearch_index(self, column, index_name=None, load=False, **kwargs):
        if load:
            self.dataset.load_elasticsearch_index(index_name=index_name, es_client=self.es_client, **kwargs)
            # fix: settings are not actually used when loading an existing ES index
            # TODO open an issue on HF to fix it upstream
            settings = kwargs.get('es_index_config', {}).get('settings')
            if settings is not None:
                es_index = self.dataset._indexes[index_name]
                es_index_name = es_index.es_index_name
                self.es_client.indices.close(es_index_name)
                self.es_client.indices.put_settings(settings, es_index_name)
                self.es_client.indices.open(es_index_name)
        else:
            self.dataset.add_elasticsearch_index(column, index_name=index_name, es_client=self.es_client, **kwargs)


class Searcher:
    """
    Aggregates several KnowledgeBases (KBs). 
    Searches through a dataset using all the indexes of all KnowledgeBases.
    Fuses results of search with multiple indexes and compute metrics.
    """
    def __init__(self, kb_kwargs, reference_kb_path=None, reference_key='passage', request_timeout=1000, 
                 es_client_kwargs={}, fusion_kwargs={}, metrics_kwargs={}):
        self.kbs = {}
        self.metrics = {}
        # FIXME maybe check if ES is needed before instantiating client?
        # this does not require ES to run anyway
        es_client = Elasticsearch(timeout=request_timeout, **es_client_kwargs)
        # load KBs used to search and index them
        resolved_kb_paths = {}
        for kb_path, kb_kwarg in kb_kwargs.items():
            resolved_kb_path = Path(kb_path).expanduser().resolve()
            if resolved_kb_path in resolved_kb_paths:
                raise ValueError(f"'{kb_path}' and '{resolved_kb_paths[resolved_kb_path]}' resolve to the same path")
            resolved_kb_paths[resolved_kb_path] = kb_path

            kb = KnowledgeBase(kb_path, es_client=es_client, **kb_kwarg)
            self.kbs[kb_path] = kb
            # same as kb.dataset._indexes.keys()
            index_names = kb.indexes.keys()
            assert not (index_names & self.metrics.keys()), "All KBs should have unique index names"
            # N. B. dict.fromkeys creates pointers to the SAME object (Counter instance here)
            self.metrics.update({index_name: Counter() for index_name in index_names})
        assert not ({'search', 'fusion'} & self.metrics.keys()), "'search', 'fusion' are reserved names"
        if len(self.metrics) > 1:
            self.do_fusion = True
            self.metrics["fusion"] = Counter()
        else:
            self.do_fusion = False

        # no reference KB
        if reference_kb_path is None:
            warnings.warn("Didn't get a reference KB "
                          "-> will not be able to extend the annotation coverage "
                          "so results should be interpreted carefully.\n")
            self.reference_kb = None
        # reference KB already loaded in KBs used to search
        elif reference_kb_path in self.kbs and self.kbs[reference_kb_path].dataset is not None:
            self.reference_kb = self.kbs[reference_kb_path].dataset
        # reference-only KB (not used to search) so we have to load it
        else:
            self.reference_kb = load_from_disk(kb_path)
        # N. B. the 'reference_kb' term is not so appropriate
        # it is not an instance of KnowledgeBase but Dataset !
        self.reference_key = reference_key
        self.fusion_method = fusion_kwargs.pop('method', 'interpolation')
        self.fusion_kwargs = fusion_kwargs
        self.metrics_kwargs = metrics_kwargs
    
    def __call__(self, batch, k=100):
        """Search using all indexes of all KBs registered in self.kbs"""
        for kb in self.kbs.values():
            for index_name, index in kb.indexes.items():
                queries = batch[index.key]
                # N. B. cannot use `None in queries` because 
                # "The truth value of an array with more than one element is ambiguous."
                if any(query is None for query in queries):
                    scores_batch, indices_batch = kb.search_batch_if_not_None(index_name, queries, k=k)
                else:
                    scores_batch, indices_batch = kb.search_batch(index_name, queries, k=k)
                # indices might need to be mapped so that all KBs refer to the same semantic index
                scores_batch, indices_batch = kb.map_indices(scores_batch, indices_batch, k=k)

                # eventually normalize the scores before fusing
                if index.normalization is not None:
                    scores_batch = normalize(scores_batch, **index.normalization)

                # store result in the dataset
                batch[f'{index_name}_scores'] = scores_batch
                batch[f'{index_name}_indices'] = indices_batch

                # are the retrieved documents relevant ?
                if self.reference_kb is not None:
                    relevant_batch = find_relevant_batch(indices_batch, batch['output'], 
                                                         self.reference_kb, reference_key=self.reference_key,
                                                         relevant_batch=deepcopy(batch['provenance_indices']))
                else:
                    relevant_batch = batch['provenance_indices']

                # compute metrics
                compute_metrics(self.metrics[index_name],
                                retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                                K=k, scores_batch=scores_batch, **self.metrics_kwargs)

        # fuse the results of the searches
        if self.do_fusion:
            self.fuse_and_compute_metrics(batch, k=k)
        return batch

    def fuse_and_compute_metrics(self, batch, k=100):
        scores_batch, indices_batch = self.fuse(batch, k=k)
        batch['search_scores'], batch['search_indices'] = scores_batch, indices_batch

        # are the retrieved documents relevant ?
        if self.reference_kb is not None:
            relevant_batch = find_relevant_batch(indices_batch, batch['output'], 
                                                 self.reference_kb, reference_key=self.reference_key,
                                                 relevant_batch=deepcopy(batch['provenance_indices']))
        else:
            relevant_batch = batch['provenance_indices']

        # compute metrics
        compute_metrics(self.metrics["fusion"],
                        retrieved_batch=indices_batch, relevant_batch=relevant_batch,
                        K=k, scores_batch=scores_batch, **self.metrics_kwargs)

        return batch

    def fuse(self, batch, k=100):
        """Should return a (scores, indices) tuples the same way as Dataset.search_batch"""
        fusions = dict(interpolation=self.interpolation_fusion)
        return fusions[self.fusion_method](batch, k=k, **self.fusion_kwargs)

    def union_results(self, batch):
        """make union of all search results"""
        batch_size = len(next(iter(batch.values())))

        all_indices = [set() for _ in range(batch_size)]
        for kb in self.kbs.values():
            for index_name, index in kb.indexes.items():
                batch_indices = batch[f'{index_name}_indices']
                for i, indices in enumerate(batch_indices):
                    all_indices[i] |= set(indices)
        return all_indices

    def interpolation_fusion(self, batch, k=100, default_minimum=False):
        """
        Simple weighted sum, e.g. : fusion = w_1*score_1 + w_2*score_2 + w_3*score_3
        The *default-minimum trick* is used in Ma et al. (2021, arXiv:2104.05740): 
        when combining results from systems A and B, it consists in giving the minimum score of A's results 
        if a given passage was only retrieved by system B, and vice-versa.

        Parameters
        ----------
        batch: dict
            as parsed by datasets
        k: int, optional
            Defaults to 100
        default_minimum: bool, optional
            Use the *default-minimum trick* (defaults to not to).
        """
        all_indices = self.union_results(batch)
        
        # init scores
        scores_dicts = [{i: 0. for i in indices} for indices in all_indices]

        for kb in self.kbs.values():
            for index_name, index in kb.indexes.items():
                weight = index.interpolation_weight
                assert weight is not None, \
                    "You should set 'interpolation_weight' for each index to use interpolation_fusion"
                # search results using index (computed previously)
                index_scores_dicts = scores2dict(batch[f'{index_name}_scores'], batch[f'{index_name}_indices'])
                # iterate over *all* retrieved indices (-> passages), not only those retrieved using this index
                for indices, scores_dict, index_scores_dict in zip(all_indices, scores_dicts, index_scores_dicts):
                    # can happen, e.g. for a face index when no face was detected
                    if not index_scores_dict:
                        continue
                    # follow Ma et al. (2021, arXiv:2104.05740) by using minimal score 
                    # when the document was retrieved only by one system
                    min_index_score = min(index_scores_dict.values()) if default_minimum else 0.
                    for i in indices:
                        score = index_scores_dict.get(i, min_index_score)
                        scores_dict[i] += weight * score

        scores_batch, indices_batch = dict_batch2scores(scores_dicts, k=k)
        return scores_batch, indices_batch


def dataset_search(dataset, k=100, metric_save_path=None, map_kwargs={}, **kwargs):
    searcher = Searcher(**kwargs)

    # HACK: sleep until elasticsearch is good to go
    time.sleep(60)

    # search expects a batch as input
    dataset = dataset.map(searcher, fn_kwargs=dict(k=k), batched=True, **map_kwargs)

    metrics = searcher.metrics
    reduce_metrics(metrics, K=k)
    print(stringify_metrics(metrics, tablefmt='latex', floatfmt=".3f"))
    if metric_save_path is not None:
        with open(metric_save_path, 'w') as file:
            json.dump(metrics, file)

    return dataset


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

    dataset = dataset_search(dataset, k,
                             metric_save_path=args['--metrics'],
                             **config)

    dataset.save_to_disk(dataset_path)
