"""
Trainee is a nn.Module that computes the loss and returns it either as:
    - first element of a tuple 
    - in the loss key of a dict or ModelOutput
so it is compatible with Trainer.
Exception is made for BiEncoder that simply wraps both encoders outputs
so it is compatible with DPRBiEncoderTrainer (and subclasses).
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import partial

import torch.nn as nn
import torch
from transformers.models.dpr.modeling_dpr import DPRReaderOutput
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_outputs import QuestionAnsweringModelOutput, ModelOutput
from transformers import BertForQuestionAnswering

from .losses import _calc_mml


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


@dataclass 
class DPRBiEncoderOutput(BiEncoderOutput):
    """
    Outputs from the question and context encoders 
    (same as DPRQuestionEncoderOutput, DPRContextEncoderOutput with prefixes)
    """
    question_pooler_output: Optional[torch.FloatTensor] = None
    question_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_attentions: Optional[Tuple[torch.FloatTensor]] = None
    context_pooler_output: Optional[torch.FloatTensor] = None
    context_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    context_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BiEncoder(nn.Module):
    """    
    Parameters
    ----------
    question_model, context_model: nn.Module
    """
    supports_gradient_checkpointing = True
    def __init__(self, question_model, context_model):
        super().__init__()
        self.question_model = question_model
        self.context_model = context_model

    def forward(self, question_inputs, context_inputs):
        """        
        Parameters
        ----------
        question_inputs, context_inputs: dict
            passed to the respective encoder
        """
        # embed questions and contexts
        question_outputs = self.question_model(**question_inputs)
        context_outputs = self.context_model(**context_inputs)

        return BiEncoderOutput(
            question_pooler_output=question_outputs.pooler_output,
            context_pooler_output=context_outputs.pooler_output)
    
    # gradient checkpointing: taken from transformers
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value
            
    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())
    

class DPRBiEncoder(BiEncoder):
    """Adapted from https://github.com/facebookresearch/DPR/blob/main/dpr/models/biencoder.py"""
    def __init__(self, question_model, context_model):
        """
        Parameters
        ----------
        question_model: transformers.DPRQuestionEncoder
            Encoder based on BERT used to encode the question/query
        context_model: transformers.DPRContextEncoder  
            Encoder based on BERT used to encode the context/evidence/passage 
            ('context' is confusing IMO but I keep it for consistency with DPR and transformers)
        """
        super().__init__(question_model=question_model, context_model=context_model)
    
    def forward(self, question_inputs, context_inputs):
        """
        Embeds questions and contexts with their respective model and returns the embeddings.
        
        * N - number of questions in a batch
        * M - number of passages per questions
        * L - sequence length
        * d - dimension of the model/embeddings
        
        Parameters
        ----------
        question_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N, L)
            usual BERT inputs, see transformers.DPRQuestionEncoder
        context_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N*M, L)
            usual BERT inputs, see transformers.DPRContextEncoder
        """
        question_outputs = self.question_model(**question_inputs)
        context_outputs = self.context_model(**context_inputs)

        return DPRBiEncoderOutput(
            question_pooler_output=question_outputs.pooler_output,
            question_hidden_states=question_outputs.hidden_states,
            question_attentions=question_outputs.attentions,
            context_pooler_output=context_outputs.pooler_output,
            context_hidden_states=context_outputs.hidden_states,
            context_attentions=context_outputs.attentions)


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
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=ignored_index)

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


class MultiPassageBERT(BertForQuestionAnswering):
    """
    PyTorch/Transformers implementation of Multi-passage BERT [1]_ (based on the global normalization [2]_)
    i.e. groups passages per question before computing the softmax (and the NLL loss)
    so that spans scores are comparable across passages

    Code based on transformers.BertForQuestionAnswering, dpr.models.Reader
    and https://github.com/allenai/document-qa/blob/master/docqa/nn/span_prediction.py

    N. B. differences with DPRReaderForQuestionAnswering:
        * no projection layer between BERT and QA-extraction
        * no re-ranking (TODO implement MultiPassageDPRReader?)
        * global normalization

    References
    ----------
    .. [1] Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 
       2019. Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering. 
       In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing 
       and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 
       pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.

    .. [2] Christopher Clark and Matt Gardner. 2018. Simple and Effective Multi-Paragraph Reading Comprehension. 
       In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 
       pages 845–855, Melbourne, Australia. Association for Computational Linguistics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self,
                input_ids,
                start_positions=None, end_positions=None, answer_mask=None,
                return_dict=None, **kwargs):
        """
        notations: 
           * N - number of distinct questions
           * M - number of passages per question in a batch
           * L - sequence length

        Parameters
        ----------
        input_ids: Tensor[int]
            shape (N * M, L)
            There should always be a constant number of passages (relevant or not) per question
        start_positions, end_positions: Tensor[int], optional
            shape (N, M, max_n_answers)
            The answer might be found several time in the same passage, maximum `max_n_answers` times
            Defaults to None (i.e. don’t compute the loss)
        answer_mask: Tensor[int], optional
            shape (N, M, max_n_answers)
            Used to mask the loss for answers that are not `max_n_answers` times in the passage
            Required if start_positions and end_positions are specified
        **kwargs: additional arguments are passed to BERT after being reshape like 
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, return_dict=True, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # compute loss
        total_loss, start_log_probs, end_log_probs = None, None, None
        if start_positions is not None and end_positions is not None:
            n_times_m, L = input_ids.size()
            M = start_positions.size(1)
            assert n_times_m % M == 0
            N = n_times_m//M
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = L
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.NLLLoss(reduction='none', ignore_index=ignored_index)

            # reshape from (N * M, L) to (N, M * L) so that all M passages related to the same question
            # will share the same softmax normalization
            start_logits, end_logits = start_logits.view(N, M*L), end_logits.view(N, M*L)
            start_log_probs, end_log_probs = self.log_softmax(start_logits), self.log_softmax(end_logits)
            # after computing the softmax, reshape back to (N * M, L)
            # because the last dimension, L, must match the position indices (i.e. class label) in start_positions, end_positions
            start_log_probs, end_log_probs = start_log_probs.view(N*M, L), end_log_probs.view(N*M, L)
            start_logits, end_logits = start_logits.view(N*M, L), end_logits.view(N*M, L)

            # reshape to match model output
            start_positions, end_positions = start_positions.view(N*M, -1), end_positions.view(N*M, -1)
            answer_mask = answer_mask.to(device=input_ids.device, dtype=torch.float32).view(N*M, -1)

            # compute span loss for each answer position in passage (in range `max_n_answers`)
            start_losses = [(loss_fct(start_log_probs, _start_positions) * _span_mask)
                            for (_start_positions, _span_mask)
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

            end_losses = [(loss_fct(end_log_probs, _end_positions) * _span_mask)
                          for (_end_positions, _span_mask)
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                          torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

            # keep the maximum per passage for each question
            loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
            total_loss = _calc_mml(loss_tensor)

        if not return_dict:
            output = (start_logits, end_logits, start_log_probs, end_log_probs) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiPassageBERTOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            start_log_probs=start_log_probs,
            end_log_probs=end_log_probs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
