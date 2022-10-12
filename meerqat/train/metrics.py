"""Metrics to be used in trainer."""
import warnings

# TODO https://torchmetrics.readthedocs.io/en/stable/retrieval/mrr.html

def retrieval(eval_prediction, ignore_index=-100):
    """
    Computes metric for retrieval training (at the batch-level)
    
    Parameters
    ----------
    eval_prediction: EvalPrediction (dict-like)
        predictions: np.ndarray
            shape (dataset_size, N*M)
            This corresponds to the log-probability of the relevant passages per batch (N*M == batch size)
        label_ids: np.ndarray
            shape (dataset_size, )
            Label at the batch-level (each value should be included in [0, N-1] inclusive)
    ignore_index: int, optional
        Labels with this value are not taken into account when computing metrics.
        Defaults to -100
    """
    warnings.warn("Not implemented. Returning empty dict.")
    metrics = {}    
    return metrics

    log_probs = eval_prediction.predictions
    dataset_size, N_times_M = log_probs.shape
    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    mrr, ignored_predictions = 0, 0
    for ranking, label in zip(rankings, eval_prediction.label_ids):
        if label == ignore_index:
            ignored_predictions += 1
            continue
        # +1 to count from 1 instead of 0
        rank = (ranking == label).nonzero()[0].item() + 1
        mrr += 1/rank
    mrr /= (dataset_size-ignored_predictions)
    metrics["MRR@N*M"] = mrr

    # argmax to get index of prediction (equivalent to `log_probs.argmax(axis=1)`)
    predictions = rankings[:, 0]
    # hits@1
    where = eval_prediction.label_ids != ignore_index
    metrics["hits@1"] = (predictions[where] == eval_prediction.label_ids[where]).mean()

    return metrics
