"""Loss functions."""
import torch

def _calc_mml(loss_tensor):
    """Taken from dpr.models.reader to avoid extra-dependency"""
    marginal_likelihood = torch.sum(torch.exp(- loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood + torch.ones(loss_tensor.size(0), device=marginal_likelihood.device) * (marginal_likelihood == 0).float()))