"""Misc. utility functions."""
from typing import Optional, Tuple
import math

import torch
from torch import nn
import numpy as np

from transformers.tokenization_utils_base import BatchEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TanhGate(nn.Module):
    """
    Flamingo-style tanh gating (init at 0) [1]_
    
    References
    ----------
    .. [1] Jean-Baptiste Alayrac et al. (2022). 
       Flamingo: a Visual Language Model for Few-Shot Learning. ArXiv:2204.14198.
    """
    def __init__(self):
        super().__init__()
        self.gate_param = nn.Parameter(torch.tensor([0.]))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        return x * self.tanh(self.gate_param)


class ResidualAttention(nn.Module):
    """
    Attention as described in Vaswani et al. without any projection, 
    i.e. the attention is computed in the input space.
    
    Includes a residual connection to be used as a drop-in replacement for BertAttention
    
    Adapted from BertSelfAttention.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False
    ) -> Tuple[torch.Tensor]:
        assert head_mask is None and past_key_value is None
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            attention_mask = encoder_attention_mask
            key = encoder_hidden_states
            value = encoder_hidden_states
        else:
            key = hidden_states
            value = hidden_states

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(hidden_states, key.transpose(-1, -2))
        scale = math.sqrt(hidden_states.shape[2])
        attention_scores = attention_scores / scale
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask.squeeze(1)
            
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        context_layer = torch.matmul(attention_probs, value)
        
        # residual connection
        output = context_layer + hidden_states
        outputs = (output, attention_probs) if output_attentions else (output,)
        return outputs


def map_if_not_None(values, function, *args, default_value=None, **kwargs):
    """
    Map all not None values through function (along with additionnal arguments)

    Values that are None will output ``default_value``

    Parameters
    ----------
    values: list
        of len batch_size
    function: callable
    default_value: optional
        Defaults to None
    *args, **kwargs: 
        additionnal arguments are passed to function

    Returns
    -------
    Output: list
        of len batch_size (same as values), with ``default_value`` where values are None
    """
    # 1. filter out values that are None
    output = []
    not_None_values, not_None_values_indices = [], []
    for i, value in enumerate(values):
        # will be overwritten for not_None_values
        output.append(default_value)
        if value is not None:
            not_None_values.append(value)
            not_None_values_indices.append(i)
    if not not_None_values:
        return output
        
    # 2. map values that are not None to function
    not_None_output = function(not_None_values, *args, **kwargs)

    # 3. return the results in a list of list with proper indices
    for j, i in enumerate(not_None_values_indices):
        output[i] = not_None_output[j]
    return output


def debug_shape(batch, prefix=""):
    """Recursively prints the shape of Tensor and ndarray in nested dict/BatchEncoding/tuple/list"""
    for key, data in batch.items():
        if isinstance(data, (dict, BatchEncoding)):
            debug_shape(data, prefix=f"{prefix}.{key}")
        elif isinstance(data, (tuple, list)):
            for i, v in enumerate(data):
                debug_shape(v, prefix=f"{prefix}.{key}.{i}")
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            print(f"{prefix}.{key} ({type(data)}): {data.shape}")
        else:
            print(f"{prefix}.{key} ({type(data)})")


def prepare_inputs(data):
    """
    Moves tensors in data to ``device``, be it a tensor or a nested list/dictionary of tensors.
    Adapted from transformers.Trainer
    """
    if isinstance(data, (dict, BatchEncoding)):
        # N. B. BatchEncoding does not accept kwargs, not sure what happens in Trainer
        return dict(**{k: prepare_inputs(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        raise TypeError(f"Unexpected type '{type(data)}' for data:\n{data}")
    return data