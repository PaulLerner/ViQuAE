# -*- coding: utf-8 -*-
"""Dataclasses for model outputs."""
from typing import Optional, Tuple
from dataclasses import dataclass

import torch

from transformers.modeling_outputs import QuestionAnsweringModelOutput, ModelOutput


@dataclass
class MultiPassageBERTOutput(QuestionAnsweringModelOutput):
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
    Same as DPRQuestionEncoderOutput / DPRContextEncoderOutput
    """
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass 
class BiEncoderOutput(ModelOutput):
    """Simply wraps both encoders output in one."""
    question_pooler_output: Optional[torch.FloatTensor] = None
    context_pooler_output: Optional[torch.FloatTensor] = None