import warnings

import numpy as np


def get_best_spans(start_probs, end_probs, weights=None, cannot_be_first_token=True):
    """
    Get the best scoring spans from start and end probabilities

    notations:
        N - number of distinct questions
        M - number of passages per question in a batch
        L - sequence length

    Parameters
    ----------
    start_probs, end_probs: ndarray
        shape (N, M, L)
    weights: ndarray, optional
        shape (N, M)
        Used to weigh the spans scores, e.g. might be BM25 scores from the retriever
    cannot_be_first_token: bool, optional
        (Default) null out the scores of start/end in the first token
        (e.g. "[CLS]", used during training for irrelevant passages)

    Returns
    -------
    passage_indices: ndarray
        shape (N, )
    start_indices, end_indices: ndarray
        shape (N, )
        start (inclusive) and end (exclusive) index of each span
    """
    N, M, L = start_probs.shape

    # 1. compute pairwise scores -> shape (N, M, L, L)
    pairwise = np.expand_dims(start_probs, -1) @ np.expand_dims(end_probs, -2)
    # fix scores where end < start
    pairwise = np.triu(pairwise)
    # null out the scores of start in the first token (and thus end because of the upper triangle)
    # (e.g. [CLS], used during training for irrelevant passages)
    if cannot_be_first_token:
        pairwise[:, :, 0, :] = 0
    # eventually weigh the scores
    if weights is not None:
        minimum = weights.min()
        if minimum < 1:
            warnings.warn("weights should be > 1, adding 1-minimum")
            weights += 1-minimum
        pairwise *= np.expand_dims(weights, (2, 3))

    # 2. find the passages with the maximum score
    pairwise = pairwise.reshape(N, M, L * L)
    max_per_passage = pairwise.max(axis=2)
    passage_indices = max_per_passage.argmax(axis=1)
    # TODO isn't there a better way to do this ?
    pairwise_best_passages = np.concatenate([np.expand_dims(p[i], 0) for i, p in zip(passage_indices, pairwise)], axis=0)

    # 3. finally find the best spans for each question
    flat_argmaxes = pairwise_best_passages.argmax(axis=-1)
    # convert from flat argmax to line index (start) and column index (end)
    start_indices = flat_argmaxes // L
    # add +1 to make end index exclusive so the spans can easily be used with slices
    end_indices = (flat_argmaxes % L) + 1

    return passage_indices, start_indices, end_indices


def format_predictions_for_squad(predictions, references):
    predictions_squad, references_squad = [], []
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        i = str(i)
        predictions_squad.append(dict(prediction_text=prediction, id=i))
        answers = dict(answer_start=[], text=[])
        for answer in reference:
            # not sure why 'answer_start' is mandatory but not used when computing the metric
            answers['answer_start'].append(0)
            answers['text'].append(answer)
        references_squad.append(dict(answers=answers, id=i))
    return predictions_squad, references_squad
