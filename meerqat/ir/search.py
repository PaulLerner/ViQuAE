"""
Script and classes to search. Built upon datasets (itself wrapping FAISS and ElasticSearch).

Usage:
search.py <dataset> <config> [--k=<k> --disable_caching --metrics=<path>]

Positional arguments:
    1. <dataset>   Path to the dataset  
    2. <config>    Path to the JSON configuration file (passed as kwargs)
    
Options:
    --k=<k>                 Hyperparameter to search for the k nearest neighbors [default: 100].
    --disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
    --metrics=<path>        Path to the directory to save the results of the run and evaluation
"""
import warnings

from docopt import docopt
import json
from collections import Counter
import time
import re
import enum
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch
from datasets import load_from_disk, set_caching_enabled
from datasets.search import ElasticSearchIndex, FaissIndex
import ranx

from .metrics import find_relevant_batch
from ..data.utils import json_integer_keys


def scores2dict(scores_batch, indices_batch):
    """
    Zips a batch of scores and indices into a batch of dict.
    
    Parameters
    ----------
    scores_batch: List[List[float]]
    indices_batch: List[List[int]]
    
    Returns
    -------
    scores_dicts: List[dict[int, float]]
    """
    scores_dicts = []
    for scores, indices in zip(scores_batch, indices_batch):
        scores_dicts.append(dict(zip(indices, scores)))
    return scores_dicts


def dict2scores(scores_dict, k=100):
    """
    Sorts in desc. order and keeps top-k.
    
    Parameters
    ----------
    scores_dict: dict[int, float]
    
    Returns
    -------
    scores: List[float]
    indice: List[int]
    """
    indices = sorted(scores_dict, key=scores_dict.get, reverse=True)[:k]
    scores = [scores_dict[index] for index in indices]
    return scores, indices


def dict_batch2scores(scores_dicts, k=100):
    """Reverses ``scores2dict``. Additionnally sorts in desc. order and keeps top-k."""
    scores_batch, indices_batch = [], []
    for scores_dict in scores_dicts:
        scores, indices = dict2scores(scores_dict, k=k)
        scores_batch.append(scores)
        indices_batch.append(indices)
    return scores_batch, indices_batch


def norm_mean_std(scores_batch, mean, std):
    """
    Scales input data.
    
    Parameters
    ----------
    scores_batch: List[List[float]]
    mean: float
    std: float
    
    Returns
    -------
    scores_batch: List[np.ndarray]
    """
    return [(np.array(scores)-mean)/std for scores in scores_batch]


def normalize(scores_batch, method, **kwargs):
    """
    Switch function for different normalization methods.
    
    Parameters
    ----------
    scores_batch: List[List[float]]
    method: str
        Name of the normalization method
    **kwargs:
        Passed to the normalization method
    
    Returns
    -------
    scores_batch
    """
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
    Dataclass to hold information about an index (either FaissIndex or ESIndex)
    
    Parameters
    ----------
    key: str
        Associated key in the dataset where the queries are stored
    kind_str: str, optional
        One of IndexKind
    es: bool, optional
        Linked to an ESIndex or FaissIndex
    do_L2norm: bool, optional
        Whether to apply ``L2norm`` to the queries
    normalization: str, optional
        If not None, applies this kind of ``normalize`` to the results scores
    interpolation_weight: float, optional
        Used to fuse the results of multiple Indexes
        
    Notes
    -----
    Difficult to create a hierarchy like FaissIndex and ESIndex since public methods, 
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
    """
    A KB can be indexed by several indexes.
    
    Parameters
    ----------
    kb_path: str, optional
        Path to the Dataset holding the KB
    index_mapping_path: str, optional
        Path to the JSON file mapping KB articles to its corresponding passages indices
    index_kwargs: dict, optional
        Each key identifies an Index and each value is passed to ``add_or_load_index``
    es_client: Elasticsearch, optional
    load_dataset: bool, optional
        This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp)
    """
    def __init__(self, kb_path=None, index_mapping_path=None, index_kwargs={}, 
                 es_client=None, load_dataset=True):
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
        """Filters out queries that are None and runs ``search_batch`` for the rest."""
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

        Notes
        -----
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
                for i in range(len(self.index_mapping[index])):
                    # add a slightly decreasing number so that the order of passage in document is preserved
                    # this is to prevent the downstream buggy sort of ranx, see https://github.com/AmenRa/ranx/issues/9
                    new_scores.append(score-i*1e-6)
                if k is not None and len(new_indices) >= k:
                    break
            new_scores_batch.append(new_scores)
            new_indices_batch.append(new_indices)
        return new_scores_batch, new_indices_batch

    def add_or_load_index(self, column=None, index_name=None, es=False, kind_str=None, key=None,
                          normalization=None, interpolation_weight=None, **index_kwarg):
        """
        Calls either ``add_or_load_elasticsearch_index`` or ``add_or_load_faiss_index``according to es.
        Unless column is None, then it does not actually add the index. 
        This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp).
        
        Parameters
        ----------
        column: str, optional
            Name/key of the column that holds the pre-computed embeddings.
        index_name: str, optional
            Index identifier. Defaults to ``column``
        es: bool, optional
        kind_str, key, normalization, interpolation_weight: 
            see Index
        **index_kwarg:
            Passed to ``add_or_load_elasticsearch_index`` or ``add_or_load_faiss_index``
        """
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
        """
        Parameters
        ----------
        column, index_name: 
            see add_or_load_index
        load: bool, optional
            Whether to ``load_faiss_index`` or ``add_faiss_index``
        save_path: str, optional
            Save index using ``self.dataset.save_faiss_index``
            Defaults not to save.
        string_factory: str, optional
            see ``Dataset.add_faiss_index`` and https://github.com/facebookresearch/faiss/wiki/The-index-factory
        device: int, optional
            see ``Dataset.add_faiss_index``
        **kwargs:
            Passed to ``load_faiss_index`` or ``add_faiss_index``
        
        Returns
        -------
        do_L2norm: bool
            Inferred from string_factory. See Index.
        """
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
        """
        When loading, it will also check the settings and eventually update them (using put_settings)
        
        Parameters
        ----------
        column, index_name: 
            see add_or_load_index
        load: bool, optional
            Whether to ``load_elasticsearch_index`` or ``add_elasticsearch_index``
        **kwargs:
            Passed to ``load_elasticsearch_index`` or ``add_elasticsearch_index``
        """
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


def format_run_indices(indices_batch, scores_batch):
    """Identifiers in ranx should be str, not int. Also, list cannot be empty because of Numba"""
    str_indices_batch, non_empty_scores = [], []
    for indices, scores in zip(indices_batch, scores_batch):
        if len(indices) > 0:
            str_indices_batch.append(list(map(str, indices)))
            non_empty_scores.append(scores)
        else:
            str_indices_batch.append(["DUMMY_RUN"])
            non_empty_scores.append([0])
    return str_indices_batch, non_empty_scores


def format_qrels_indices(indices_batch):
    """Identifiers in ranx should be str, not int. Also, list cannot be empty because of Numba"""
    str_indices_batch, non_empty_scores = [], []
    for indices in indices_batch:
        if len(indices) > 0:
            str_indices_batch.append(list(map(str, indices)))
            # relevance score should always be 1, see https://github.com/AmenRa/ranx/issues/5
            non_empty_scores.append([1 for _ in indices])
        else:
            str_indices_batch.append(["DUMMY_QREL"])
            non_empty_scores.append([0])
    return str_indices_batch, non_empty_scores


class Searcher:
    """
    Aggregates several KnowledgeBases (KBs). 
    Searches through a dataset using all the indexes of all KnowledgeBases.
    Fuses results of search with multiple indexes and compute metrics.
    
    Parameters
    ----------
    kb_kwargs: dict
        Each key identifies a KB and each valye is passed to KnowledgeBase
    k: int, optional
        Searches for the top-k results
    reference_kb_path: str, optional
        Path to the Dataset that hold the reference KB, used to evaluate the results.
        If it is one of self.kbs, it will only get loaded once.
        Defaults to evaluate only from the cached 'provenance_indices' in the dataset (not recommanded).
    reference_key: str,
        Used to get the reference field in kb
        Defaults to 'passage'
    request_timeout: int, optional
        Timeout for Elasticsearch
    es_client_kwargs: dict, optional
        Passed to Elasticsearch
    fusion_kwargs: dict, optional
        Passed to the fusion method (see fuse). Default method is interpolation_fusion.
    metrics_kwargs: dict, optional
        Passed to ranx.compare. Defaults to "mrr", "precision", "hit_rate" at ranks [1, 5, 10, 20, 100]
    save_in_dataset: bool, optional
        Whether to save the dataset after searching (which will hold the results)
        or rely only on ranx to save.
    """
    def __init__(self, kb_kwargs, k=100, reference_kb_path=None, reference_key='passage', request_timeout=1000,
                 es_client_kwargs={}, fusion_kwargs={}, metrics_kwargs={}, save_in_dataset=True):
        self.k = k
        self.kbs = {}
        self.qrels = ranx.Qrels()
        self.runs = {}
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
            assert not (index_names & self.runs.keys()), "All KBs should have unique index names"
            for index_name in index_names:
                run = ranx.Run()
                run.name = index_name
                self.runs[index_name] = run
        assert not ({'search', 'fusion'} & self.runs.keys()), "'search', 'fusion' are reserved names"
        if len(self.runs) > 1:
            self.do_fusion = True
            run = ranx.Run()
            run.name = "fusion"
            self.runs["fusion"] = run
            if not save_in_dataset:
                save_in_dataset = True
                warnings.warn("setting save_in_dataset=True to be able to fuse results")
        else:
            self.do_fusion = False
        self.save_in_dataset = save_in_dataset

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
            self.reference_kb = load_from_disk(reference_kb_path)
        # N. B. the 'reference_kb' term is not so appropriate
        # it is not an instance of KnowledgeBase but Dataset !
        self.reference_key = reference_key
        self.fusion_method = fusion_kwargs.pop('method', 'interpolation')
        self.fusion_kwargs = fusion_kwargs
        # I advise against using any kind of metric that uses recall (mAP, R-Precision, …) since we estimate
        # relevant document on the go so the number of relevant documents will *depend on the systemS* you use
        ks = metrics_kwargs.pop("ks", [1, 5, 10, 20, 100])
        default_metrics_kwargs = dict(metrics=[f"{m}@{k}" for m in ["mrr", "precision", "hit_rate"] for k in ks])
        default_metrics_kwargs.update(metrics_kwargs)
        self.metrics_kwargs = default_metrics_kwargs
    
    def __call__(self, batch):
        """Search using all indexes of all KBs registered in self.kbs"""
        for kb in self.kbs.values():
            for index_name, index in kb.indexes.items():
                queries = batch[index.key]
                # N. B. cannot use `None in queries` because 
                # "The truth value of an array with more than one element is ambiguous."
                if any(query is None for query in queries):
                    scores_batch, indices_batch = kb.search_batch_if_not_None(index_name, queries, k=self.k)
                else:
                    scores_batch, indices_batch = kb.search_batch(index_name, queries, k=self.k)
                # indices might need to be mapped so that all KBs refer to the same semantic index
                scores_batch, indices_batch = kb.map_indices(scores_batch, indices_batch, k=self.k)

                # eventually normalize the scores before fusing
                if index.normalization is not None:
                    scores_batch = normalize(scores_batch, **index.normalization)

                # store result in the dataset
                if self.save_in_dataset:
                    batch[f'{index_name}_scores'] = scores_batch
                    batch[f'{index_name}_indices'] = indices_batch

                # store results in the run
                str_indices_batch, non_empty_scores = format_run_indices(indices_batch, scores_batch)
                self.runs[index_name].add_multi(
                    q_ids=batch['id'],
                    doc_ids=str_indices_batch,
                    scores=non_empty_scores
                )

                # are the retrieved documents relevant ?
                if self.reference_kb is not None:
                    # extend relevant documents with the retrieved
                    # /!\ this means you should not compute/interpret recall as it will vary depending on the run/system
                    find_relevant_batch(indices_batch, batch['output'], self.reference_kb,
                                        reference_key=self.reference_key, relevant_batch=batch['provenance_indices'])

        # fuse the results of the searches
        if self.do_fusion:
            self.fuse_and_compute_metrics(batch)

        # add Qrels scores depending on the documents retrieved by the systems
        str_indices_batch, non_empty_scores = format_qrels_indices(batch['provenance_indices'])
        self.qrels.add_multi(
            q_ids=batch['id'],
            doc_ids=str_indices_batch,
            scores=non_empty_scores
        )
        return batch

    def fuse_and_compute_metrics(self, batch):
        """First calls fuse, then find_relevant_batch."""
        scores_batch, indices_batch = self.fuse(batch)
        batch['search_scores'], batch['search_indices'] = scores_batch, indices_batch

        # store results in the run
        str_indices_batch, non_empty_scores = format_run_indices(indices_batch, scores_batch)
        self.runs["fusion"].add_multi(
            q_ids=batch['id'],
            doc_ids=str_indices_batch,
            scores=non_empty_scores
        )

        # are the retrieved documents relevant ?
        if self.reference_kb is not None:
            # extend relevant documents with the retrieved
            # /!\ this means you should not compute/interpret recall as it will vary depending on the run/system
            find_relevant_batch(indices_batch, batch['output'], self.reference_kb,
                                reference_key=self.reference_key, relevant_batch=batch['provenance_indices'])

        return batch

    def fuse(self, batch):
        """Should return a (scores, indices) tuples the same way as Dataset.search_batch"""
        fusions = dict(interpolation=self.interpolation_fusion)
        return fusions[self.fusion_method](batch, **self.fusion_kwargs)

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

    def interpolation_fusion(self, batch, default_minimum=False):
        """
        Simple weighted sum, e.g. : fusion = w_1*score_1 + w_2*score_2 + w_3*score_3
        The *default-minimum trick* is used in [1]_ 
        when combining results from systems A and B, it consists in giving the minimum score of A's results 
        if a given passage was only retrieved by system B, and vice-versa.

        Parameters
        ----------
        batch: dict
            as parsed by datasets
        default_minimum: bool, optional
            Use the *default-minimum trick* (defaults to not to).
            
        References
        ----------
        .. [1] Ma, X., Sun, K., Pradeep, R., & Lin, J. (2021). A replication study of dense passage retriever. 
           arXiv preprint arXiv:2104.05740. https://arxiv.org/abs/2104.05740
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

        scores_batch, indices_batch = dict_batch2scores(scores_dicts, k=self.k)
        return scores_batch, indices_batch


def dataset_search(dataset, k=100, metric_save_path=None, map_kwargs={}, **kwargs):
    """
    Instantiates searcher, maps the dataset through it, then compute and saves metrics.
    
    Parameters
    ----------
    dataset: Dataset
    k: int, optional
        see Searcher
    metric_save_path: str, optional
        Path to the directory where to save the results qrels, runs and metrics of eval_dataset.
        Defaults not to save.
    map_kwargs: dict, optional
        Passed to self.dataset.map
    **kwargs:
        Passed to Searcher
    """
    searcher = Searcher(k=k, **kwargs)

    # HACK: sleep until elasticsearch is good to go
    time.sleep(60)

    # search expects a batch as input
    dataset = dataset.map(searcher, batched=True, **map_kwargs)

    # compute metrics
    report = ranx.compare(
        searcher.qrels,
        runs=searcher.runs.values(),
        **searcher.metrics_kwargs
    )

    print(report)
    # save qrels, metrics (in JSON and LaTeX), statistical tests, and runs.
    if metric_save_path is not None:
        metric_save_path.mkdir(exist_ok=True)
        searcher.qrels.save(metric_save_path/"qrels.trec", kind='trec')
        report.save(metric_save_path/"metrics.json")
        with open(metric_save_path/"metrics.tex", 'wt') as file:
            file.write(report.to_latex())
        for index_name, run in searcher.runs.items():
            run.save(metric_save_path/f"{index_name}.trec", kind='trec')
    
    if searcher.save_in_dataset:
        return dataset
    return None


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
                             metric_save_path=Path(args['--metrics']),
                             **config)
    if dataset is not None:
        dataset.save_to_disk(dataset_path)
