"""
Usage:
metrics.py relevant <dataset> <passages> <title2index> <article2passage> [--disable_caching]
"""
from docopt import docopt
import json
from tabulate import tabulate
import warnings
import re

import numpy as np
from datasets import load_from_disk

from meerqat.data.loading import answer_preprocess
from meerqat.data.utils import json_integer_keys


def find_relevant(retrieved, original_answer, alternative_answers, kb, reference_key='passage'):
    """
    Parameters
    ----------
    retrieved: List[int]
    original_answer: str
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


def find_relevant_batch(retrieved_batch, ground_truth_output_batch, kb, relevant_batch=None, reference_key='passage', original_answer_only=False):
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
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(find_relevant_item, fn_kwargs=kwargs)
    dataset.save_to_disk(dataset_path)


def get_irrelevant_batch(retrieved_batch, relevant_batch):
    irrelevant_batch = []
    for retrieved, relevant in zip(retrieved_batch, relevant_batch):
        # N. B. list because sets are not allowed in datasets
        irrelevant_batch.append(list(set(retrieved) - set(relevant)))
    return irrelevant_batch


def compute_metrics(metrics, retrieved_batch, relevant_batch, K=100, ks=[1, 5, 10, 20, 100], scores_batch=None):
    """
    Parameters
    ----------
    metrics: Counter
        to store the results
    retrieved_batch: List[List[int]]
        Indices of the retrieved documents for all queries in the batch
    relevant_batch: List[List[int]]
        Indices of the ground-truth documents for all queries in the batch
    K: int, optional
        Number of documents queried to the system (default: 100)
        Maximum threshold for the R in R-Precision and the k in, e.g. precision@k
    ks: List[int], optional
        Used for, e.g. precision@k
        Cannot be greater than K
        Defaults to [1, 5, 10, 20, 100]
    scores_batch: List[List[float]], optional
        scores of retrieved documents for all queries in the batch
        (not used in any metric for now)
    """
    for retrieved, relevant in zip(retrieved_batch, relevant_batch):
        metrics["total_queries"] += 1
        if len(relevant) == 0:
            metrics["no_ground_truth"] += 1
            continue

        relevant_set = set(relevant)

        # R-Precision
        R = min(len(relevant_set), K)
        metrics[f"r-precision@{K}"] += len(set(retrieved[:R]) & relevant_set)/R

        # Reciprocal Rank
        for rank, index in enumerate(retrieved):
            if index in relevant_set:
                metrics[f"MRR@{K}"] += 1/(rank+1)
                break            

        # (hits|precision|recall)@k
        for k in ks:
            # do not compute, e.g. precision@100 if you only asked for 10 documents
            if k > K:
                continue
            retrieved_at_k = retrieved[:k]
            retrieved_set = set(retrieved_at_k)
            relret_at_k = len(retrieved_set & relevant_set)
            metrics[f"hits@{k}"] += min(1, relret_at_k)
            metrics[f"precision@{k}"] += relret_at_k/k
            metrics[f"recall@{k}"] += relret_at_k/R


def reduce_metrics(metrics_dict, K=100, ks=[1, 5, 10, 20, 100]):
    for key, metrics in metrics_dict.items():
        no_ground_truth = metrics.get("no_ground_truth", 0)
        if no_ground_truth > 0:
            warnings.warn(f"{no_ground_truth} queries out of {metrics['total_queries']} had no-ground truth but are still taken into account.")
        if metrics["total_queries"] <= 0:
            continue

        # average MRR, r-precision, hits, precision and recall over the whole dataset
        for metric in [f"r-precision@{K}", f"MRR@{K}"]:
            metrics[metric] /= metrics["total_queries"]
        for k in ks:
            if k > K:
                continue
            for metric in [f"hits@{k}", f"precision@{k}", f"recall@{k}"]:
                metrics[metric] /= metrics["total_queries"]
        metrics_dict[key] = metrics
    return metrics_dict


def stringify_metrics(metrics_dict, **kwargs):
    string_list = []
    for key, metrics in metrics_dict.items():
        string_list.append(key)
        string_list.append(tabulate([metrics], headers='keys', **kwargs))
        string_list.append('\n\n')
    return '\n'.join(string_list)


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['relevant']:
        passages = load_from_disk(args['<passages>'])
        with open(args['<title2index>'], 'r') as file:
            title2index = json.load(file)
        with open(args['<article2passage>'], 'r') as file:
            article2passage = json.load(file, object_hook=json_integer_keys)
        find_relevant_dataset(args['<dataset>'], passages=passages, title2index=title2index, article2passage=article2passage)

