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


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['relevant']:
        passages = load_from_disk(args['<passages>'])
        with open(args['<title2index>'], 'r') as file:
            title2index = json.load(file)
        with open(args['<article2passage>'], 'r') as file:
            article2passage = json.load(file, object_hook=json_integer_keys)
        find_relevant_dataset(args['<dataset>'], passages=passages, title2index=title2index, article2passage=article2passage)

