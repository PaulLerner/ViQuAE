from tabulate import tabulate

def find_relevant(retrieved_passages, answers):
    """"""
    raise NotImplementedError


def compute_metrics(metrics, retrieved_batch, relevant_batch, K=100, ks=[1, 5, 10, 20, 100], scores=None):
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
    ks: List[int], optional
        Used for, e.g. precision@k
        Cannot be greater than K
        Defaults to [1, 5, 10, 20, 100]
    scores: List[List[float], optional
        scores of retrieved documents for all queries in the batch
        (not used in any metric for now)
    """
    for retrieved, relevant in zip(retrieved_batch, relevant_batch):
        metrics["total_queries"] += 1
        relevant_set = set(relevant)

        # R-Precision
        R = len(relevant)
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
            retrieved_set = set(retrieved)
            relret_at_k = len(retrieved_set & relevant_set)
            metrics[f"hits@{k}"] += min(1, relret_at_k)
            metrics[f"precision@{k}"] += relret_at_k/k
            metrics[f"recall@{k}"] += relret_at_k/R


def reduce_metrics(metrics_dict, K=100, ks=[1, 5, 10, 20, 100]):
    for key, metrics in metrics_dict.items():
        # average MRR, r-precision, hits, precision and recall over the whole dataset over the whole dataset
        for metric in [f"r-precision@{K}", f"MRR@{K}"]:
            metrics[metric] /= metrics["total_queries"]
        for k in ks:
            for metric in ["r-precision", f"hits@{k}", f"precision@{k}", f"recall@{k}"]:
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