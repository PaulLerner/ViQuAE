"""Metrics to be used in trainer."""
import warnings
import re
import string
from collections import Counter                                                                                                                                                                                    

from ..data.loading import answer_preprocess


# TODO https://torchmetrics.readthedocs.io/en/stable/retrieval/mrr.html
def retrieval(eval_outputs, ignore_index=-100):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_outputs: List[dict[str, Tensor]]
        Contains log_probs and labels for all batches in the evaluation step (either validation or test)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """
    metrics = {}    
    mrr, hits_at_1, ignored_predictions, dataset_size = 0, 0, 0, 0
    for batch in eval_outputs:
        log_probs = batch['log_probs'].detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
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
    """

    assert len(predictions) == len(references)
    f1, exact_match = 0, 0
    for prediction, ground_truths in zip(predictions, references):
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        breakpoint()

    exact_match = 100.0 * exact_match / len(references)
    f1 = 100.0 * f1 / len(references)

    return {"exact_match": exact_match, "f1": f1}

