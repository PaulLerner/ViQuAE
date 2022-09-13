"""
Script and functions related to metrics and ranx.

(for docopt) Usage:
metrics.py relevant <dataset> <passages> <title2index> <article2passage> [--disable_caching]
metrics.py qrels <qrels>... --output=<path>
metrics.py ranx --qrels=<path> [<run>... --output=<path> --filter=<path> --kwargs=<path> --cats=<path>]
metrics.py (win|tie|loss) <metrics> [--metric=<metric>]
                                  
Usages:
    1. metrics.py relevant <dataset> <passages> <title2index> <article2passage> [--disable_caching]    
    2. metrics.py qrels <qrels>... --output=<path>    
    3. metrics.py ranx --qrels=<path> [<run>... --output=<path> --filter=<path> --kwargs=<path> --cats=<path>]     
    4. metrics.py (win|tie|loss) <metrics> [--metric=<metric>]  
    
Positional arguments:
    * <usage>              Pick one usage.
    * <dataset>            Path to the dataset  
    * <passages>           Path to the passages (also a Dataset)
    * <title2index>        Path to the JSON file mapping article’s title to it’s index in the KB
    * <article2passage>    Path to the JSON file mapping article’s index to its corresponging passage indices
    * <qrels>...           Paths to the Qrels to merge
    * <metrics>            Path to the JSON metrics file (output of ranx)

Options:
    --disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
    --output=<path>         1. qrels: output path to the TREC file
                            2. ranx: output path to the directory where to save metrics
    --filter=<path>         Path towards the JSON file that contains a list of question ids to filter *out*
    --kwargs=<path>         Path towards the JSON config file that contains kwargs
    --cats=<path>           Path towards the JSON that maps categories to their question ids
    --metric=<metric>       Metric on which to compute wins/ties/losses [default: precision@1].                                                            
"""

from docopt import docopt
import json
import warnings
import re
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import load_from_disk
import ranx

from ..data.loading import answer_preprocess
from ..data.utils import json_integer_keys


def find_relevant(retrieved, original_answer, alternative_answers, kb, reference_key='passage'):
    """
    Parameters
    ----------
    retrieved: List[int]
    original_answer: str
        Included in alternative_answers so original_relevant is included in relevant
    alternative_answers: List[str]
    kb: Dataset
    reference_key: str, optional
        Used to get the reference field in kb
        Defaults to 'passage'

    Returns
    -------
    original_relevant, relevant: List[int]
        Included in retrieved
    """
    original_relevant, relevant = [], []
    for i in retrieved:
        i = int(i)
        passage = answer_preprocess(kb[i][reference_key])

        answer = answer_preprocess(original_answer)
        if re.search(rf'\b{answer}\b', passage) is not None:
            original_relevant.append(i)
            relevant.append(i)
            continue

        for answer in alternative_answers:
            answer = answer_preprocess(answer)
            if re.search(rf'\b{answer}\b', passage) is not None:
                relevant.append(i)
                break
    return original_relevant, relevant


def find_relevant_batch(retrieved_batch, ground_truth_output_batch, kb, 
                        relevant_batch=None, reference_key='passage', original_answer_only=False):
    """
    Applies ``find_relevant`` on a batch (output of ir.search)
    
    Parameters
    ----------
    retrieved_batch: List[List[int]]
    ground_truth_output_batch: List[dict]
    kb: Dataset
    relevant_batch: List[List[int]], optional
        Each item in relevant_batch will be extended with the other relevant indices we find with ``find_relevant``
        Defaults to a batch of empty lists.
    reference_key: str, optional
        Used to get the reference field in kb
        Defaults to 'passage'
    original_answer_only: bool, optional
        Consider that only the original answer is relevant, not alternative ones.
    """
    if relevant_batch is None:
        batch_size = len(ground_truth_output_batch)
        relevant_batch = [[] for _ in range(batch_size)]

    for retrieved, relevant, ground_truth_output in zip(retrieved_batch, relevant_batch, ground_truth_output_batch):
        # we already know that relevant indices are relevant, no need to compute it twice
        retrieved_todo = set(retrieved) - set(relevant)
        if original_answer_only:
            alternative_answers = []
        else:
            alternative_answers = ground_truth_output['answer']
        _, r = find_relevant(retrieved_todo, ground_truth_output['original_answer'], alternative_answers, kb, reference_key=reference_key)
        relevant.extend(r)

    return relevant_batch


def find_relevant_item(item, passages, title2index, article2passage):
    """
    Applies ``find_relevant`` with passages of articles linked to the question.
    
    Parameters
    ----------
    item: dict
    passages: Dataset
    title2index: dict[str, int]
        Mapping article’s title to it’s index in the KB
    article2passage: dict[int, List[int]]
        Mapping article’s index to its corresponging passage indices
    """
    # ignore from which paragraph the answer comes from
    # (might have been quicker to do this mapping in make_passage)
    titles = set(provenance['title'][0] for provenance in item['output']['provenance'])
    original_relevant, relevant = [], []
    for title in titles:
        if title not in title2index:
            continue
        article_index = title2index[title]
        passage_indices = article2passage.get(article_index, [])
        o, r = find_relevant(passage_indices, item['output']['original_answer'], item['output']['answer'], passages)
        original_relevant.extend(o)
        relevant.extend(r)
    item['original_answer_provenance_indices'] = original_relevant
    item['provenance_indices'] = relevant
    return item


def find_relevant_dataset(dataset_path, **kwargs):
    """Loads dataset, maps it through find_relevant_item and saves it back."""
    dataset = load_from_disk(dataset_path)
    # TODO save qrels in TREC/ranx format in dataset_path/qrels.trec
    dataset = dataset.map(find_relevant_item, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def fuse_qrels(qrels_paths):
    """
    Loads all qrels in qrels_paths and unions them under a single Qrels.
    
    Parameters
    ----------
    qrels_paths: List[str]
    
    Returns
    -------
    fused_qrels: ranx.Qrels
    """
    # nothing to fuse
    if len(qrels_paths) == 1:
        return ranx.Qrels.from_file(qrels_paths[0], kind='trec')
    final_qrels = {}
    for qrels_path in tqdm(qrels_paths):
        qrels = ranx.Qrels.from_file(qrels_path, kind='trec').qrels
        for q_id, rels in qrels.items():
            final_qrels.setdefault(q_id, {})
            for doc_id, score in rels.items():
                final_qrels[q_id].setdefault(doc_id, {})
                final_qrels[q_id][doc_id] = score
    return ranx.Qrels.from_dict(final_qrels)


def load_runs(runs_paths=[], runs_dict={}, filter_q_ids=[]):
    """
    Loads runs from both run_paths and runs_dict. Eventually filters out some questions.
    
    Parameters
    ----------
    runs_paths: List[str], optional
    runs_dict: dict[str, str], optional
        {name of the run: path of the run}
    filter_q_ids: List[str]
        Question identifiers to filter from the runs
        
    Returns
    -------
    runs: List[ranx.Run]
    """
    runs = []
    
    # load runs from CLI
    for run_path in runs_paths:
        run = ranx.Run.from_file(run_path, kind='trec')
        if run.name is None:
            run.name = run_path
        else:
            run.name += run_path
        for q_id in filter_q_ids:
            run.run.pop(q_id)
        runs.append(run)
    
    # load runs from config file
    for name, run_path in runs_dict.items():
        run = ranx.Run.from_file(run_path, kind='trec')
        run.name = name
        for q_id in filter_q_ids:
            run.run.pop(q_id)
        runs.append(run)
    
    return runs
        

def compare(qrels_path, runs_paths=[], runs_dict={}, output_path=None, filter_q_ids=[], **kwargs):
    """
    Loads Qrels and Runs, feed them to ranx.compare and save result.
    
    Parameters
    ----------
    qrels_path: str
    runs_paths: List[str], optional
    runs_dict: dict[str, str], optional
        {name of the run: path of the run}
    output_path: str, optional
        Path of the directory were to save output JSON and TeX files.
        Defaults not to save (only print results)
    filter_q_ids: List[str]
        Question identifiers to filter from the Runs and Qrels
    **kwargs:
        Passed to ranx.compare
    """
    qrels = ranx.Qrels.from_file(qrels_path, kind='trec')
    for q_id in filter_q_ids:
        qrels.qrels.pop(q_id)
    
    runs = load_runs(runs_paths, runs_dict=runs_dict, filter_q_ids=filter_q_ids)

    report = ranx.compare(
        qrels,
        runs=runs,
        **kwargs
    )
    print(report)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        report.save(output_path / "metrics.json")
        with open(output_path / "metrics.tex", 'wt') as file:
            file.write(report.to_latex())


def cat_breakdown(qrels_path, runs_paths, cats, runs_dict={}, output_path=None, 
                  filter_q_ids=[], metrics=["mrr"]):
    """
    qrels_path, runs_paths, runs_dict, output_path, filter_q_ids: 
        see ``compare``
    cats: dict[str, List[str]]
        {category: list of question identifiers that belong to it}
    metrics: List[str], optional
        Which metrics to compute
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
    qrels = ranx.Qrels.from_file(qrels_path, kind='trec')    
    runs = load_runs(runs_paths, runs_dict=runs_dict)
    
    # break qrels by cat
    qrels_by_cat = {}
    for cat, q_ids in cats.items():
        qrels_by_cat[cat] = ranx.Qrels({q_id: qrels.qrels[q_id] for q_id in q_ids})
    
    # break runs by cat
    runs_by_cat = []
    for run in runs:
        run_by_cat = {}
        for cat, q_ids in cats.items():
            run_by_cat[cat] = ranx.Run({q_id: run.run[q_id] for q_id in q_ids}, name=run.name)
        runs_by_cat.append(run_by_cat)
            
    # compute metrics for each cat
    for metric in metrics:
        metric_by_cat = {}
        for cat, qrels_of_cat in qrels_by_cat.items():
            for run_by_cat in runs_by_cat:
                run = run_by_cat[cat]
                metric_by_cat.setdefault(run.name, {})
                #TODO use compare instead of evaluate and print report with stat test
                metric_by_cat[run.name][cat] = ranx.evaluate(qrels_of_cat, run, metric)
        
        df = pd.DataFrame(metric_by_cat)
        means = df.mean()
        df = df.T
        df['micro-avg'] = means
        print(metric)
        print(df.to_latex(float_format='{:,.1%}'.format))
        print('\n***********\n')
        if output_path is not None:
            df.to_csv(output_path/f'{metric}.csv')
    

def get_wtl_table(metrics, wtl_key='W', wtl_metric='precision@1'):
    """
    Formats either the wins, ties, or losses of the models against each other
    according to wtl_key in a pandas.DataFrame
    
    metrics: dict
        loaded from the JSON output of ranx
    wtl_key: str, optional
        Whether to compute the win ('W'), tie ('T'), or loss ('L')
    wtl_metric: str, optional
        What does it mean to win?
    """
    for k in ["metrics", "model_names", "stat_test"]:
        metrics.pop(k, None)
    table = {}
    for model, metric in metrics.items():
        table[model] = {model:0}
        for m2, wtl in metric['win_tie_loss'].items():
            table[model][m2] = wtl[wtl_metric][wtl_key]    
    return pd.DataFrame(table).T
    

if __name__ == '__main__':
    args = docopt(__doc__)
    wtl_key = None
    
    if args['relevant']:
        passages = load_from_disk(args['<passages>'])
        with open(args['<title2index>'], 'r') as file:
            title2index = json.load(file)
        with open(args['<article2passage>'], 'r') as file:
            article2passage = json.load(file, object_hook=json_integer_keys)
        find_relevant_dataset(args['<dataset>'], passages=passages, title2index=title2index, article2passage=article2passage)
    
    elif args['qrels']:
        qrels = fuse_qrels(args['<qrels>'])
        qrels.save(args['--output'], kind='trec')
        
    elif args['ranx']:
        # usage: either cat_breakdown or compare
        if args['--cats'] is not None:
            with open(args['--cats'], 'rt') as file:
                cats = json.load(file)
        else:
            cats = None            
        if args['--filter'] is not None:
            with open(args['--filter'], 'rt') as file:
                filter_q_ids = json.load(file)
        else:
            filter_q_ids = []
        if args['--kwargs'] is not None:
            with open(args['--kwargs'], 'rt') as file:
                kwargs = json.load(file)
        else:
            ks = [1, 5, 10, 20, 100]
            kwargs = dict(metrics=[f"{m}@{k}" for m in ["precision", "mrr"] for k in ks])
        if args['<run>'] is not None:
            runs_paths = args['<run>']
        else:
            runs_paths = []        
        if cats is None:
            compare(args['--qrels'], runs_paths, output_path=args['--output'], 
                    filter_q_ids=filter_q_ids, **kwargs)
        else:            
            cat_breakdown(args['--qrels'], runs_paths, output_path=args['--output'], 
                          cats=cats, filter_q_ids=filter_q_ids, **kwargs)
                    
    elif args['win']:
        wtl_key = 'W'
    elif args['tie']:
        wtl_key = 'T'
    elif args['loss']:
        wtl_key = 'L'
    if wtl_key is not None:
        metric = args['--metric']
        if metric is None:
            metric = 'precision@1'
        with open(args['<metrics>'], 'rt') as file:
            metrics = json.load(file)
        wtl = get_wtl_table(metrics, wtl_key=wtl_key, wtl_metric=metric)
        print(wtl.to_latex())

