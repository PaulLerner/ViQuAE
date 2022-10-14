"""Loss functions, optimizers, and schedulers."""
import torch
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
    # it might still depend on M, the number of passages, because there’s a max-pooling in MultiPassageBERT
    return -torch.mean(torch.log(marginal_likelihood + torch.ones(loss_tensor.size(0), device=marginal_likelihood.device) * (marginal_likelihood == 0).float()))