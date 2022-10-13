"""Metrics to be used in trainer."""
import warnings
    

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
