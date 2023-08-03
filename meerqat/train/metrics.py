"""Metrics to be used in trainer."""
import warnings
from collections import Counter                                                                                                                                                                                    

import ranx

from ..data.loading import answer_preprocess
    

def accumulate_batch_metrics(batch_metrics):    
    metrics = Counter()   
    for metric in batch_metrics:
        for k, v in metric.items():
            metrics[k] += v
    effective_size = metrics.pop("batch_size") - metrics.pop("ignored_predictions", 0)
    for k, v in metrics.items():
        metrics[k] = v/effective_size
    return metrics


# TODO https://torchmetrics.readthedocs.io/en/stable/retrieval/mrr.html
def batch_retrieval(log_probs, labels, ignore_index=-100):
    mrr, hits_at_1, ignored_predictions = 0, 0, 0
    batch_size, _ = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    for ranking, label in zip(rankings, labels):
        if label == ignore_index:
            ignored_predictions += 1
            continue
        if ranking[0] == label:
            hits_at_1 += 1
        # +1 to count from 1 instead of 0
        rank = (ranking == label).nonzero()[0].item() + 1
        mrr += 1/rank    
    return {"MRR@N*M": mrr, "hits@1": hits_at_1, 
            "ignored_predictions": ignored_predictions, "batch_size": batch_size}


def retrieval(eval_outputs, ignore_index=-100, output_key='log_probs'):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains log_probs and labels for all batches in the evaluation step (either validation or test)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    output_key: str, optional
        Name of the model output in eval_outputs
    """
    metrics = {}    
    mrr, hits_at_1, ignored_predictions, dataset_size = 0, 0, 0, 0
    for batch in eval_outputs:
        log_probs = batch[output_key].numpy()
        labels = batch['labels'].numpy()
        batch_size, _ = log_probs.shape
        dataset_size += batch_size
        # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
        rankings = (-log_probs).argsort(axis=1)
        for ranking, label in zip(rankings, labels):
            if label == ignore_index:
                ignored_predictions += 1
                continue
            if ranking[0] == label:
                hits_at_1 += 1
            # +1 to count from 1 instead of 0
            rank = (ranking == label).nonzero()[0].item() + 1
            mrr += 1/rank    
    metrics["MRR@N*M"] = mrr / (dataset_size-ignored_predictions)
    metrics["hits@1"] = hits_at_1 / (dataset_size-ignored_predictions)
    return metrics


def get_run(eval_outputs, ir_run):
    """    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains logits for all batches in the evaluation step (either validation or test)
    ir_run: ranx.Run
        Original IR run being re-ranked.
    """
    run = {}
    for batch in eval_outputs:
        logits = batch['logits'].numpy()
        N, M = logits.shape
        question_ids = [batch['ids'][i] for i in range(0, N*M, M)]
        rankings = (-logits).argsort(axis=1)
        for ranking, logit, question_id in zip(rankings, logits, question_ids):
            ir_results = ir_run.run[question_id]
            # nothing to re-rank. 
            # this can happen if searching for something unavailable in the query
            # e.g. no face was detected but you are searching for face similarity (see ir.search)
            if not ir_results:
                run[question_id] = ir_results
            else:
                doc_ids = list(ir_results.keys())[: M]
                run[question_id] = {doc_ids[i]: logit[i] for i in ranking}
    return ranx.Run(run)


def f1_score(prediction, ground_truth):
    prediction_tokens = answer_preprocess(prediction).split()
    ground_truth_tokens = answer_preprocess(ground_truth).split()                                                                                                                                                  
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return answer_preprocess(prediction) == answer_preprocess(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def squad(predictions, references):
    """
    Adapted from datasets.load_metric('squad')
    
    Parameters
    ----------
    predictions: List[str]
    references: List[List[str]]
    
    Returns
    -------
    metrics: dict[str, float]
    """

    assert len(predictions) == len(references)
    f1, exact_match = 0, 0
    for prediction, ground_truths in zip(predictions, references):
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / len(references)
    f1 = f1 / len(references)

    return {"exact_match": exact_match, "f1": f1}


def squad_per_question(predictions, references):
    """
    Returns the score of the metrics for each question instead of averaging like squad.
    Keep different implementation because squad should in principle be loaded from datasets.
    This should allow for stastitical significant testing downstream.
    
    Parameters
    ----------
    predictions: List[str]
    references: List[List[str]]
    
    Returns
    -------
    metrics: dict[str, List[float]]
    """

    assert len(predictions) == len(references)
    f1, exact_match = [], []
    for prediction, ground_truths in zip(predictions, references):
        exact_match.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
        f1.append(metric_max_over_ground_truths(f1_score, prediction, ground_truths))

    return {"exact_match": exact_match, "f1": f1}

