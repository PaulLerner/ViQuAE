from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch
from transformers.models.dpr.modeling_dpr import DPRReaderOutput

from meerqat.train.losses import _calc_mml


class Trainee(nn.Module):
    """Base class for all Trainee models (to be trained by a Trainer)
    Should implement a forward function that returns loss between output and target (as a tuple, dict or ModelOutput)
    The actual forward pass should be done using the model attribute
    """
    def __init__(self, model):
        super().__init__()
        self.model = model


@dataclass
class DPRReaderForQuestionAnsweringOutput(DPRReaderOutput):
    """Same as DPRReaderOutput with an extra loss attribute (or as QuestionAnsweringModelOutput with relevance_logits)

    N. B. unfortunately we have to redefine everything so that loss is the first attribute
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    relevance_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class DPRReaderForQuestionAnswering(Trainee):
    def forward(self,
                input_ids, attention_mask,
                start_positions=None, end_positions=None, answer_mask=None,
                return_dict=None, **kwargs):
        """Based on transformers.BertForQuestionAnswering and dpr.models.Reader"""
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        outputs = self.model(input_ids, attention_mask, return_dict=True, **kwargs)

        # compute loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.view(N * M, -1)
            end_positions = end_positions.view(N * M, -1)
            answer_mask = answer_mask.view(N * M, -1)
            start_logits, end_logits, relevance_logits = outputs[:3]
            start_logits = start_logits.view(N * M, -1)
            end_logits = end_logits.view(N * M, -1)
            relevance_logits = relevance_logits.view(N * M)

            answer_mask = answer_mask.to(device=relevance_logits.device, dtype=torch.float32)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # compute switch loss
            relevance_logits = relevance_logits.view(N, M)
            switch_labels = torch.zeros(N, dtype=torch.long, device=relevance_logits.device)
            switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))

            # compute span loss
            start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask)
                            for (_start_positions, _span_mask)
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

            end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask)
                          for (_end_positions, _span_mask)
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                          torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

            loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
            span_loss = _calc_mml(loss_tensor)
            total_loss = span_loss + switch_loss

        if not return_dict:
            outputs = outputs.to_tuple()
            return ((total_loss,) + outputs) if total_loss is not None else outputs

        return DPRReaderForQuestionAnsweringOutput(loss=total_loss, **outputs)

