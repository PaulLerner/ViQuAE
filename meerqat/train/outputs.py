# -*- coding: utf-8 -*-
"""Dataclasses for model outputs."""
from typing import Optional
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
class BiEncoderOutput(ModelOutput):
    """Simply wraps both encoders output in one."""
    question_pooler_output: Optional[torch.FloatTensor] = None
    context_pooler_output: Optional[torch.FloatTensor] = None