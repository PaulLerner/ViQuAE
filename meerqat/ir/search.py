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
import re
from pathlib import Path
import enum

import numpy as np
try:
    from elasticsearch import Elasticsearch
except ImportError as e:
    warnings.warn(f"ImportError: {e}")
try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError as e:
    warnings.warn(f"ImportError: {e}")
from datasets import load_from_disk, set_caching_enabled
from datasets.search import ElasticSearchIndex, FaissIndex
import ranx

from .metrics import find_relevant
from .fuse import Fusion
from ..data.utils import json_integer_keys
from ..data.infoseek import QuestionType


def L2norm(queries):
    """Normalize each query to have a unit-norm. Expects a batch of vectors of the same dimension"""
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    return queries/norms


class IndexKind(enum.Enum):
    FAISS = 0
    ES = 1
    PYSERINI = 2
    
    
class Index:
    """
    Dataclass to hold information about an index (either FaissIndex or ESIndex)
    
    Parameters
    ----------
    key: str
        Associated key in the dataset where the queries are stored
    kind: IndexKind, optional
    do_L2norm: bool, optional
        Whether to apply ``L2norm`` to the queries
        
    Notes
    -----
    Difficult to create a hierarchy like FaissIndex and ESIndex since public methods, 
    such as search_batch, are defined in Dataset and take as input the index name.
    """
    def __init__(self, key, kind=IndexKind.FAISS, do_L2norm=False):
        self.key = key
        self.kind = kind
        self.do_L2norm = do_L2norm
        # TODO replace do_L2norm by
     #   In [43]:         projected = kb._indexes['sscd_disc_mixup'].faiss_index.sa_encode.sa_encode(foo)
    #...:         projected = np.frombuffer(projected, dtype=np.float32).reshape(1,512)


class KnowledgeBase:
    """
    A KB can be indexed by several indexes.
    
    Parameters
    ----------
    kb_path: str, optional
        Path to the Dataset holding the KB
    index_mapping_path: str, optional
        Path to the JSON file mapping KB articles to its corresponding passages indices
    many2one: str, optional
        strategy to apply in case of many2one mapping (e.g. multiple passages to article)
        Choose from {'max'}. Has no effect if index_mapping_path is None.
        Defaults assume that mapping is one2many (e.g. article to multiple passages) 
        so it will overwrite results in iteration order if it is not the case.
    index_kwargs: dict, optional
        Each key identifies an Index and each value is passed to ``add_or_load_index``
    es_client: Elasticsearch, optional
    load_dataset: bool, optional
        This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp)
    """
    def __init__(self, kb_path=None, index_mapping_path=None, many2one=None, index_kwargs={}, 
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
        self.many2one = many2one
        for index_name, index_kwarg in index_kwargs.items():
            self.add_or_load_index(index_name=index_name, **index_kwarg)
    
    def pyserini_search_batch(self, index_name, queries, k=100, threads=10):
        qids = [str(i) for i in range(len(queries))]
        results_batch = self.indexes[index_name].searcher.batch_search(queries, qids, k=k, threads=threads)
        scores_batch, indices_batch = [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]
        for i, results in results_batch.items():
            i = int(i)
            for result in results:
                scores_batch[i].append(result.score)
                indices_batch[i].append(result.docid)
        return scores_batch, indices_batch            
            
    def search_batch(self, index_name, queries, k=100):
        """Pre-process queries according to index before computing self.dataset.search_batch"""
        index = self.indexes[index_name]
        # search through pyserini or datasets
        if index.kind == IndexKind.PYSERINI:
            return self.pyserini_search_batch(index_name, queries, k=k)
        # N. B. should be equivalent to isinstance(self.dataset._indexes[index_name], FaissIndex)        
        elif index.kind == IndexKind.FAISS:
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

    def add_or_load_index(self, column=None, index_name=None, kind=None, key=None, **index_kwarg):
        """
        Calls either ``add_or_load_elasticsearch_index`` or ``add_or_load_faiss_index``according to es.
        Unless column is None, then it does not actually add the index. 
        This is useful for hyperparameter search if you want to use pre-computed results (see ir.hp).
        
        Parameters
        ----------
        column: str
            Name/key of the column that holds the pre-computed embeddings.
        index_name: str, optional
            Index identifier. Defaults to ``column``
        kind: IndexKind, optional
        **index_kwarg:
            Passed to ``add_or_load_elasticsearch_index`` or ``add_or_load_faiss_index``
        """
        if kind is None:
            kind = IndexKind.FAISS
        else:
            kind = IndexKind[kind]
        if index_name is None:
            index_name = column
        if kind==IndexKind.ES:
            do_L2norm = False
            self.indexes[index_name] = Index(key=key, kind=kind, do_L2norm=do_L2norm)
            self.add_or_load_elasticsearch_index(column, index_name=index_name, **index_kwarg)
        elif kind==IndexKind.PYSERINI:
            do_L2norm = False            
            self.indexes[index_name] = Index(key=key, kind=kind, do_L2norm=do_L2norm)
            self.add_or_load_pyserini_index(column, index_name=index_name, **index_kwarg)
        else:                
            do_L2norm = self.add_or_load_faiss_index(column, index_name=index_name, **index_kwarg)
            self.indexes[index_name] = Index(key=key, kind=kind, do_L2norm=do_L2norm)

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
    
    def add_or_load_pyserini_index(self, column=None, index_name=None, save_path=None, k1=0.9, b=0.4):
        """
        Parameters
        ----------
        column: placeholder
        index_name: str
        save_path: str
        k1 : float
            BM25 k1 parameter. (Default from pyserini)
        b : float
            BM25 b parameter. (Default from pyserini)
        """
        if column is not None:
            warnings.warn(f"Unused parameter column={column}")
        self.indexes[index_name].searcher = LuceneSearcher(save_path)
        self.indexes[index_name].searcher.set_bm25(k1=k1, b=b)
        
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


class Searcher:
    """
    Aggregates several KnowledgeBases (KBs). 
    Searches through a dataset using all the indexes of all KnowledgeBases.
    Fuses results of search with multiple indexes and compute metrics.
    
    Parameters
    ----------
    kb_kwargs: dict
        Each key identifies a KB and each value is passed to KnowledgeBase
    k: int, optional
        Searches for the top-k results
    reference_kb_path: str, optional
        Path to the Dataset that hold the reference KB, used to evaluate the results.
        If it is one of self.kbs, it will only get loaded once.
        Defaults to evaluate only from the provided qrels (not recommanded).
    reference_key: str, optional
        Used to get the reference field in kb
        Defaults to 'passage'
    qrels: str, optional
        Path to the qrels JSON file. 
        Defaults to start looking for relevant documents from scratch in self.reference_kb
        At least one of {reference_kb_path, qrels} should be provided
    request_timeout: int, optional
        Timeout for Elasticsearch
    es_client_kwargs: dict, optional
        Passed to Elasticsearch
    fusion_kwargs: dict, optional
        Passed to Fusion (see fuse)
    metrics_kwargs: dict, optional
        Passed to ranx.compare. 
        Defaults to {"metrics": ["mrr@100", "precision@1", "precision@20", "hit_rate@20"]}
    do_fusion: bool, optional
        Whether to fuse results of the indexes. Defaults to True if their are multiple indexes.
    qnonrels: str, optional
        Path towards a JSON collection of irrelevant documents. Used as cache to make search faster.
        Defaults to look for all results.
    """
    def __init__(self, kb_kwargs, k=100, reference_kb_path=None, reference_key='passage', 
                 qrels=None, request_timeout=1000, es_client_kwargs={}, fusion_kwargs={}, 
                 metrics_kwargs={}, do_fusion=None, qnonrels=None):
        self.k = k
        self.kbs = {}
        if qrels is None:
            self.qrels = {}
        else:
            with open(qrels, 'rt') as file:
                self.qrels = json.load(file)
                
        if qnonrels is None:
            self.qnonrels = {}
        else:
            with open(qnonrels, 'rt') as file:
                self.qnonrels = json.load(file)
                
        self.runs = {}
        # FIXME maybe check if ES is needed before instantiating client?
        # this does not require ES to run anyway
        es_client = Elasticsearch("https://localhost:9200", timeout=request_timeout, **es_client_kwargs)
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
                self.runs[index_name] = {}
        assert not ({'search', 'fusion'} & self.runs.keys()), "'search', 'fusion' are reserved names"
        if do_fusion is None and len(self.runs) > 1:
            self.do_fusion = True
        else:
            self.do_fusion = do_fusion
        if self.do_fusion:
            assert len(self.runs) > 1

        # no reference KB
        if reference_kb_path is None:
            assert qrels is not None
            warnings.warn("Didn't get a reference KB "
                          "-> will not be able to extend the annotation coverage "
                          "so results should be interpreted carefully.\n")
            self.reference_kb = None
        # (re)load reference KB so we can remove columns and make find_relevant faster
        else:
            self.reference_kb = load_from_disk(reference_kb_path)
            self.reference_kb = self.reference_kb.remove_columns([c for c in self.reference_kb.column_names if c != reference_key])
            
        # N. B. the 'reference_kb' term is not so appropriate
        # it is not an instance of KnowledgeBase but Dataset !
        
        self.reference_key = reference_key
        self.fusion_kwargs = fusion_kwargs
        # I advise against using any kind of metric that uses recall (mAP, R-Precision, …) since we estimate
        # relevant document on the go so the number of relevant documents will *depend on the systemS* you use
        default_metrics_kwargs = dict(metrics=["mrr@100", "precision@1", "precision@20", "hit_rate@20"])
        default_metrics_kwargs.update(metrics_kwargs)
        self.metrics_kwargs = default_metrics_kwargs

    def __call__(self, batch):
        """Search using all indexes of all KBs registered in self.kbs"""
        question_types = [QuestionType[question_type] for question_type in batch.get('question_type', ['String']*len(batch['id']))]
        for kb in self.kbs.values():
            for index_name, index in kb.indexes.items():
                queries = batch[index.key]
                # N. B. cannot use `None in queries` because 
                # "The truth value of an array with more than one element is ambiguous."
                if any(query is None for query in queries):
                    scores_batch, indices_batch = kb.search_batch_if_not_None(index_name, queries, k=self.k)
                else:
                    scores_batch, indices_batch = kb.search_batch(index_name, queries, k=self.k)
                for q_id, scores, indices, gt, question_type in zip(batch['id'], 
                                                                    scores_batch, 
                                                                    indices_batch, 
                                                                    batch['output'], 
                                                                    question_types):
                    self.runs[index_name].setdefault(q_id, {})
                    for score, i in zip(scores, indices):
                        penalty = 0.0
                        if kb.index_mapping is not None:
                            for j in kb.index_mapping[i]:
                                j = str(j)
                                # assumes one2many mapping: simply overwrite any previous values
                                if kb.many2one is None:
                                    self.runs[index_name][q_id][j] = score - penalty
                                    penalty += 1e-8
                                # keep maximum score from many2one mapping
                                elif kb.many2one == 'max':
                                    if j not in self.runs[index_name][q_id] or self.runs[index_name][q_id][j] < score:
                                        self.runs[index_name][q_id][j] = score
                                else:
                                    raise ValueError(
                                        f"Invalid value for many2one: '{kb.many2one}'. "
                                        "Choose from {None, 'max'}"
                                    )
                        else:                 
                            self.runs[index_name][q_id][str(i)] = score
                        if len(self.runs[index_name][q_id]) >= self.k:
                            break
                    # are the retrieved documents relevant ?
                    if self.reference_kb is not None:
                        # extend relevant documents with the retrieved
                        # /!\ this means you should not compute/interpret recall as it will vary depending on the run/system
                        self.qrels.setdefault(q_id, {})                        
                        self.qnonrels.setdefault(q_id, {})
                        retrieved = self.runs[index_name][q_id].keys()-(self.qrels[q_id].keys()|self.qnonrels[q_id].keys())
                        _, relevant = find_relevant(
                            retrieved, 
                            gt['original_answer'], 
                            gt['answer'], 
                            self.reference_kb, 
                            reference_key=self.reference_key,
                            question_type=question_type
                        )
                        self.qrels[q_id].update({str(i): 1 for i in relevant})   
                        self.qnonrels[q_id].update({i: 0 for i in retrieved-self.qrels[q_id].keys()})
                        
        return batch


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

    # search expects a batch as input
    dataset = dataset.map(searcher, batched=True, **map_kwargs)
    
    # dict to ranx Qrels/Run
    searcher.qrels = ranx.Qrels(searcher.qrels)
    for name, run in searcher.runs.items():
        searcher.runs[name] = ranx.Run(run, name=name)
    
    # save qrels, metrics (in JSON and LaTeX), statistical tests, and runs.
    if metric_save_path is not None:
        metric_save_path.mkdir(exist_ok=True)
        searcher.qrels.save(metric_save_path/"qrels.json")
        if searcher.qnonrels is not None:
            with open(metric_save_path/"qnonrels.json", 'wt') as file:
                json.dump(searcher.qnonrels, file)

        for index_name, run in searcher.runs.items():
            run.save(metric_save_path/f"{index_name}.json")
    
    # compute metrics
    report = ranx.compare(
        searcher.qrels,
        runs=searcher.runs.values(),
        **searcher.metrics_kwargs
    )
    print(report)    
    
    # save metrics (in JSON and LaTeX) and statistical tests
    if metric_save_path is not None:
        report.save(metric_save_path/"metrics.json")
        with open(metric_save_path/"metrics.tex", 'wt') as file:
            file.write(report.to_latex())
        
    # fuse the results of the searches
    if searcher.do_fusion:
        subcommand = searcher.fusion_kwargs.pop('subcommand')
        subcommand_kwargs = searcher.fusion_kwargs.pop('subcommand_kwargs', {})
        fuser = Fusion(
            qrels=searcher.qrels,
            runs=list(searcher.runs.values()),        
            output=metric_save_path,
            **searcher.fusion_kwargs
        )
        getattr(fuser, subcommand)(**subcommand_kwargs)    


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
    metric_save_path = Path(args['--metrics']) if args['--metrics'] is not None else None
    dataset_search(dataset, k, metric_save_path=metric_save_path, **config)
