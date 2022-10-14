"""
Trainee is a pl.LightningModule that computes the loss so it is compatible with Trainer.
"""
from functools import partial
import re
import numpy as np

import torch.nn as nn
import torch
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertForQuestionAnswering, BertTokenizer
import pytorch_lightning as pl

from ..data.loading import get_pretrained
from ..models.mm import FlamantEncoder
from ..models.qa import get_best_spans
from .optim import _calc_mml, LinearLRWithWarmup
from .outputs import MultiPassageBERTOutput, BiEncoderOutput
from .metrics import retrieval, squad


class Trainee(pl.LightningModule):
    """
    Base class for all Trainee models (to be trained by a Trainer)
    
    Parameters
    ----------    
    *args, **kwargs: additionnal arguments are passed to pl.LightningModule
    freeze_regex: str, optional
        represents a regex used to match the model parameters to freeze
        (i.e. set ``requires_grad = False``).
        Defaults to None (keep model fully-trainable)
    gradient_checkpointing: bool, optional
    lr, eps, weight_decay: float, optional
    betas: Tuple[float], optional    
    warmup_steps: int, optional
        Defaults to no warm-up
    """
    supports_gradient_checkpointing = True
    def __init__(self, *args, freeze_regex=None, gradient_checkpointing=False,                  
                 warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_regex = freeze_regex
        self.gradient_checkpointing = gradient_checkpointing
        
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
    # should be called at the end of each subclass __init__
    def post_init(self):
        if self.freeze_regex is not None:
            self.freeze(self.freeze_regex)        
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
    def step(self, *args, **kwargs):
        raise NotImplementedError("Subclass and implement step.")
    
    def training_step(self, *args, **kwargs):
        """Step and log training metrics"""
        outputs = self.step(*args, **kwargs)
        self.log("train/loss", outputs['loss'])
        return outputs
    
    def validation_step(self, *args, **kwargs):
        """Step and log validation metrics"""
        outputs = self.step(*args, **kwargs)
        self.log("eval/loss", outputs['loss'])
        return outputs
    
    def test_step(self, *args, **kwargs):
        """Step and log test metrics"""
        outputs = self.step(*args, **kwargs)
        self.log("test/loss", outputs['loss'])
        return outputs
    
    def eval_epoch_end(self, eval_outputs):
        raise NotImplementedError("Subclass and implement eval_epoch_end.")
    
    def validation_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)
        for k, v in metrics.items():
            self.log(f"eval/{k}", v)
            
    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
            
    def get_weights_to_log(self, model):
        logs = {}
        weights_to_log = getattr(model, 'weights_to_log', {})
        for name, tensor in weights_to_log.items():
            logs[name] = tensor.cpu().detach().item()
        return logs
    
    def freeze(self, regex):
        """
        Overrides freeze to freeze only parameters that match the regex.
        Caveat: does not call .eval() so does not disable Dropout
        """
        regex = re.compile(regex)
        total, frozen = 0, 0
        print("Model parameters:\n"+"Name".ljust(120)+"\t#Trainable\t#Total")
        for name, param in self.named_parameters():
            numel = param.numel()
            if regex.match(name):
                param.requires_grad = False
                frozen += numel
            total += numel
            print(f"{name.ljust(120)}\t{(numel if param.requires_grad else 0):,d}\t{numel:,d}")
        print(f"Froze {frozen:,d} parameters out of {total:,d}")
        
    # TODO delete once I understand how to setup scheduling interval in config.yaml
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        
        total_steps=self.trainer.estimated_stepping_batches
        scheduler = LinearLRWithWarmup(
            optimizer,
            warmup_steps=self.warmup_steps, total_steps=total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    
    ###################################################
    # gradient checkpointing: taken from transformers #
    ###################################################
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BertEncoder, FlamantEncoder)):
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

        
class BiEncoder(Trainee):
    """    
    The training objective is to minimize the negative log-likelihood of the similarities (dot product)
    between the questions and the passages embeddings, as described in [3]_.
    
    References
    ----------
    .. [3] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. 
       Dense Passage Retrieval for Open-Domain Question Answering. 
       Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020.

    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to Trainee
    question_class: str
        Name of the class used for question_model. See get_class_from_name.
    question_model_name_or_path: str
        Passed to from_pretrained. See transformers.PreTrainedModel.from_pretrained
    context_class: str, optional
        Analog to question_class for context_model. Defaults to question_class.
    context_model_name_or_path: str
        Analog to question_model_name_or_path for context_model. Defaults to question_model_name_or_path.
    """
    def __init__(self, *args, question_class, question_model_name_or_path, 
                 context_class=None, context_model_name_or_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # default to symmetric encoders
        context_class = question_class if context_class is None else context_class
        context_model_name_or_path = question_model_name_or_path if context_model_name_or_path is None else context_model_name_or_path
        
        # init encoders
        self.question_model = get_pretrained(question_class, pretrained_model_name_or_path=question_model_name_or_path)
        self.context_model = get_pretrained(context_class, pretrained_model_name_or_path=context_model_name_or_path)
        
        # loss and metrics
        self.log_softmax = nn.LogSoftmax(1)
        self.loss_fct = nn.NLLLoss(reduction='mean')
        
        self.post_init()        
        
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
    
    def log_question_and_context(self):
        """Maybe add weights of question and context model to the logs"""
        logs = {}
        # TODO
        return logs
        question_logs = self.get_weights_to_log(self.question_model)
        # add "question_" prefix
        logs.update({f"question_{k}": question_logs[k] for k in list(question_logs.keys())})
        
        context_logs = self.get_weights_to_log(self.context_model)
        # add "context_" prefix
        logs.update({f"context_{k}": context_logs[k] for k in list(context_logs.keys())})

        return logs
    
    def step(self, inputs, _):
        """
        Calculates In-batch negatives schema loss and supports to run it in DDP mode 
        by exchanging the representations across all the nodes.
        
        Adapted from https://github.com/facebookresearch/DPR/blob/main/train_dense_encoder.py
        and https://github.com/Lightning-AI/lightning/discussions/14390
        
        Notes
        -----
        This means that the whole representations of questions and contexts, and their similarity matrix, must fit on a single GPU.
        """            
        local_labels = inputs.pop('labels')  # (N, )

        outputs = self(**inputs)

        global_outputs = self.all_gather(outputs, sync_grads=True)
        global_labels = self.all_gather(local_labels, sync_grads=True)

        # compute similarity
        similarities = global_outputs.question_pooler_output @ global_outputs.context_pooler_output.T  # (N, N*M)
        log_probs = self.log_softmax(similarities)

        loss = self.loss_fct(log_probs, global_labels)
        return dict(loss=loss, log_probs=log_probs, labels=global_labels)   
    
    def eval_epoch_end(self, eval_outputs):
        return retrieval(eval_outputs)
            
    
class MultiPassageBERT(BertForQuestionAnswering):
    """
    PyTorch/Transformers implementation of Multi-passage BERT [1]_ (based on the global normalization [2]_)
    i.e. groups passages per question before computing the softmax (and the NLL loss)
    so that spans scores are comparable across passages

    Code based on transformers.BertForQuestionAnswering, dpr.models.Reader
    and https://github.com/allenai/document-qa/blob/master/docqa/nn/span_prediction.py

    Notes
    -----
    Differences with DPRReaderForQuestionAnswering:
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

    def __init__(self, *args, M=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_softmax = nn.LogSoftmax(1)
        self.M = M

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
            The answer might be found several time in the same passage, maximum ``max_n_answers`` times
            Defaults to None (i.e. don’t compute the loss)
        answer_mask: Tensor[int], optional
            shape (N, M, max_n_answers)
            Used to mask the loss for answers that are not ``max_n_answers`` times in the passage
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
            n_times_m, L = input_ids.shape
            M = start_positions.shape[1]
            assert n_times_m % M == 0
            assert self.M is None or self.M == M
            self.M = M
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


class MultiPassageBERTTrainee(Trainee):
    """    
    Parameters
    ----------
    model_name_or_path: str
    tokenizer_name_or_path: str
    """
    def __init__(self, *args, model_name_or_path, tokenizer_name_or_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MultiPassageBERT.from_pretrained(model_name_or_path)
        # FIXME: tokenizer is already loaded in train.data
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.post_init()   
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep answer_strings in batch. Does not try to cast them as Tensor of any dtype or device."""
        answer_strings = batch.pop('answer_strings', None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        batch['answer_strings'] = answer_strings
        
        return batch
    
    def step(self, inputs, _):   
        keep_for_eval = {k: inputs.pop(k, None) for k in ['answer_strings', 'passage_scores']}
        model_outputs = self(**inputs)
        keep_for_eval['input_ids'] = inputs['input_ids']
        keep_for_eval.update(model_outputs)
        return keep_for_eval
    
    def log_probs_to_answers(self, start_log_probs, end_log_probs, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        passage_indices, start_indices, end_indices = get_best_spans(start_probs=np.exp(start_log_probs),
                                                                     end_probs=np.exp(end_log_probs),
                                                                     **kwargs)
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.trainer.datamodule.tokenizer.batch_decode(answers, skip_special_tokens=True)
    
    def eval_epoch_end(self, eval_outputs):        
        M = self.model.M
        # gather all outputs
        all_predictions, all_weighted_predictions, all_answer_strings = [], [], []
        dataset_size = 0
        for batch in eval_outputs:
            input_ids = batch['input_ids']
            n_times_m, L = input_ids.shape
            N = n_times_m//M
            dataset_size += N
            answer_strings = batch['answer_strings']
            answer_strings = [answer_strings[i] for i in range(0, len(answer_strings), M)]
            assert len(answer_strings) == N
            all_answer_strings.extend(answer_strings)
            
            # TODO keep in torch
            input_ids = input_ids.detach().cpu().numpy().reshape(N, M, L)
            start_log_probs = batch['start_log_probs'].detach().cpu().numpy().reshape(N, M, L)
            end_log_probs = batch['end_log_probs'].detach().cpu().numpy().reshape(N, M, L)
            
            predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids)
            all_predictions.extend(predictions)
            passage_scores = batch['passage_scores']
            if passage_scores is not None:
                passage_scores = passage_scores.detach().cpu().numpy().reshape(N, M)
                weighted_predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids, weights=passage_scores)
                all_weighted_predictions.extend(weighted_predictions)
        assert len(all_answer_strings) == dataset_size
        # compute metrics        
        metrics = squad(predictions=all_predictions, references=all_answer_strings)
        if weighted_predictions:
            for k, v in squad(predictions=all_weighted_predictions, references=all_answer_strings).items():
                metrics['weighted_'+k] = v
        return metrics
