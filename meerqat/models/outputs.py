# -*- coding: utf-8 -*-
"""Dataclasses for model outputs."""
from typing import Optional, Tuple
from dataclasses import dataclass

import torch

from transformers.modeling_outputs import QuestionAnsweringModelOutput, ModelOutput


@dataclass
class ReaderOutput(QuestionAnsweringModelOutput):
    """
    Same as QuestionAnsweringModelOutput but with start and end log-probabilities

    (equivalent to softmax(start_logits) when there is only one passage per question)
    """
    start_log_probs: torch.FloatTensor = None
    end_log_probs: torch.FloatTensor = None


@dataclass
class EncoderOutput(ModelOutput):
    """Generic class for any encoder output of the BiEncoder framework."""
    pooler_output: Optional[torch.FloatTensor] = None


@dataclass 
class ECAEncoderOutput(EncoderOutput):
    """
    Returns the full sequence hidden states (optionally across layer) 
    and attentions scores in addition to pooled sequence embedding
    """
    pooler_output: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass 
class BiEncoderOutput(ModelOutput):
    """Simply wraps both encoders output in one."""
    question_pooler_output: Optional[torch.FloatTensor] = None
    context_pooler_output: Optional[torch.FloatTensor] = None
    
    
@dataclass 
class JointMonoAndCrossModalOutput(ModelOutput):
    question_images: Optional[torch.FloatTensor] = None
    context_images: Optional[torch.FloatTensor] = None
    context_titles: Optional[torch.FloatTensor] = None
    
    
@dataclass 
class JointBiEncoderAndClipOutput(BiEncoderOutput, JointMonoAndCrossModalOutput):
    pass
    

@dataclass
class ReRankerOutput(ModelOutput):
    """

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None