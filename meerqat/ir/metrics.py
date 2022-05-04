"""
Usage:
metrics.py relevant <dataset> <passages> <title2index> <article2passage> [--disable_caching]
metrics.py qrels <qrels>... --output=<path>
metrics.py ranx --qrels=<path> [<run>... --output=<path> --filter=<path> --kwargs=<path>]
"""
from docopt import docopt
import json
import warnings
import re
from tqdm import tqdm
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import ranx

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


def fuse_qrels(qrels_paths):
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


def compare(qrels_path, runs_paths, runs_dict={}, output_path=None, filter_q_ids=[], **kwargs):
    qrels = ranx.Qrels.from_file(qrels_path, kind='trec')
    for q_id in filter_q_ids:
        qrels.qrels.pop(q_id)
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


if __name__ == '__main__':
    args = docopt(__doc__)
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
        compare(args['--qrels'], runs_paths, output_path=args['--output'], filter_q_ids=filter_q_ids, **kwargs)

