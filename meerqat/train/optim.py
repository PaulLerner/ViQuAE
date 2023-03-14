"""Loss functions, optimizers, and schedulers."""
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


class LinearLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """
    def __init__(self, *args, warmup_steps, total_steps, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)
            
    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0, 
            float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
        )


def _calc_mml(loss_tensor):
    """Taken from dpr.models.reader to avoid extra-dependency"""
    marginal_likelihood = torch.sum(torch.exp(- loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    # Mean reduction: this is different from https://github.com/facebookresearch/DPR/blob/a31212dc0a54dfa85d8bfa01e1669f149ac832b7/dpr/models/reader.py#L180
    # who use sum reduction
    # by averaging, the loss does not depend on the batch size N (number of questions)
    # it might still depend on M, the number of passages, if `max_pooling=True` in `multi_passage_rc_loss` (not recommanded)
    return -torch.mean(torch.log(marginal_likelihood + torch.ones(loss_tensor.size(0), device=marginal_likelihood.device) * (marginal_likelihood == 0).float()))


def multi_passage_rc_loss(input_ids, start_positions, end_positions, start_logits, end_logits, answer_mask, max_pooling=False):
    n_times_m, L = input_ids.shape
    M = start_positions.shape[1]
    assert n_times_m % M == 0
    N = n_times_m//M
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = L
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)
    loss_fct = nn.NLLLoss(reduction='none', ignore_index=ignored_index)
    log_softmax = nn.LogSoftmax(1)

    # reshape from (N * M, L) to (N, M * L) so that all M passages related to the same question
    # will share the same softmax normalization
    start_logits, end_logits = start_logits.view(N, M*L), end_logits.view(N, M*L)
    start_log_probs, end_log_probs = log_softmax(start_logits), log_softmax(end_logits)
    # after computing the softmax, reshape back to (N * M, L)
    # because the last dimension, L, must match the position indices (i.e. class label) in start_positions, end_positions
    start_log_probs, end_log_probs = start_log_probs.view(N*M, L), end_log_probs.view(N*M, L)
    start_logits, end_logits = start_logits.view(N*M, L), end_logits.view(N*M, L)

    # reshape to match model output
    start_positions, end_positions = start_positions.view(N*M, -1), end_positions.view(N*M, -1)
    answer_mask = answer_mask.to(device=input_ids.device, dtype=torch.float32).view(N*M, -1)

    # compute span loss for each answer position in passage (in range `max_n_answers`)
    # note that start_log_probs is constant through the loop
    start_losses = [(loss_fct(start_log_probs, _start_positions) * _span_mask)
                    for (_start_positions, _span_mask)
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

    end_losses = [(loss_fct(end_log_probs, _end_positions) * _span_mask)
                  for (_end_positions, _span_mask)
                  in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                  torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
    
    # LEGACY: keep the maximum per passage for each question
    # this might be used to reproduce the experiments of the ViQuAE paper (Lerner et al. 2022)
    # but hurts performance. see https://github.com/facebookresearch/DPR/issues/244
    if max_pooling:
        loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
    
    total_loss = _calc_mml(loss_tensor)
    
    return total_loss, start_positions, end_positions, start_logits, end_logits, start_log_probs, end_log_probs
