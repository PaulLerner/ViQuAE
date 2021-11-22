"""Metrics to be used in trainer."""

def retrieval(eval_prediction):
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
    """
    print(f"eval_prediction.predictions.shape: {eval_prediction.predictions.shape}")
    print(f"               .label_ids.shape: {eval_prediction.label_ids.shape}")
    metrics = {}

    log_probs = eval_prediction.predictions
    dataset_size, N_times_M = log_probs.shape

    # use argsort to rank the passages w.r.t. their log-probability (`-` to sort in desc. order)
    rankings = (-log_probs).argsort(axis=1)
    mrr = 0
    for ranking, label in zip(rankings, eval_prediction.label_ids):
        mrr += 1/((ranking == label).nonzero()[0].item()+1)
    mrr /= dataset_size
    metrics["MRR@N*M"] = mrr

    # argmax to get index of prediction (equivalent to `log_probs.argmax(axis=1)`)
    predictions = rankings[:,0]

    # hits@1 
    metrics["hits@1"] = (predictions==eval_prediction.label_ids).mean()

    return metrics
