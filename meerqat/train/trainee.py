"""
Trainee is a pl.LightningModule that computes the loss so it is compatible with Trainer.
"""
import warnings
from functools import partial
import re
from pathlib import Path
import json
from tqdm import tqdm

import numpy as np
import ranx

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from ..data.loading import get_pretrained
from ..data.utils import to_latex
from ..models.qa import batched_get_best_spans
from ..models.outputs import BiEncoderOutput
from .optim import LinearLRWithWarmup
from .metrics import retrieval, squad, get_run
from .data import ReRankerDataModule, pad_and_cat

    
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
        
    def step(self, batch, batch_idx):
        raise NotImplementedError("Subclass and implement step.")
    
    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)
            
    def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("train/loss", outputs['loss'])
        return outputs
    
    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("eval/loss", outputs['loss'])
        return outputs
    
    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("test/loss", outputs['loss'])
        return outputs
    
    def eval_epoch_end(self, eval_outputs):
        warnings.warn("eval_epoch_end is not implemented.")
        return {}
    
    def validation_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)['metrics']
        for k, v in metrics.items():
            self.log(f"eval/{k}", v)
            
    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        metrics = self.eval_epoch_end(*args, **kwargs)['metrics']
        print(to_latex(metrics))
        log_dir = Path(self.trainer.log_dir)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        with open(log_dir/'metrics.json', 'wt') as file:
            json.dump(metrics, file)
                
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
        
        # FIXME: this will be overwritten when loading state from ckpt_path
        # so if you want to keep training by increasing total_steps, 
        # your LR will be 0 if the ckpt reached the previously set total_steps
        total_steps=self.trainer.estimated_stepping_batches
        scheduler = LinearLRWithWarmup(
            optimizer,
            warmup_steps=self.warmup_steps, total_steps=total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    
    #####################################################
    # gradient checkpointing: adapted from transformers #
    #####################################################
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
            
    def gradient_checkpointing_enable(self):
        """
        Activates gradient checkpointing for the current model.
        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
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
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())


class CrossModal(Trainee):
    def __init__(self, *args, model_kwargs: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_pretrained(**model_kwargs)

    def step(self, batch, batch_idx):
        return self.model(**batch, return_loss=True, return_dict=True)
    
    def eval_epoch_end(self, eval_outputs):
        for batch in eval_outputs:
            batch['labels'] = torch.arange(batch['logits_per_image'].shape[1], dtype=torch.long)
        return {'metrics': retrieval(eval_outputs, output_key='logits_per_image')}

        
def _get_bert(dpr_encoder):
    if hasattr(dpr_encoder, 'question_encoder'):
        return dpr_encoder.question_encoder.bert_model
    return dpr_encoder.ctx_encoder.bert_model


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
        If 'shared', then use the same model to encode questions and passages. 
        Will set shared_encoders=True 
    context_model_name_or_path: str
        Analog to question_model_name_or_path for context_model. Defaults to question_model_name_or_path.
    """
    def __init__(self, *args, question_class, question_model_name_or_path, 
                 context_class=None, context_model_name_or_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO pass kwargs to question/context models
        
        # default to symmetric encoders
        context_class = question_class if context_class is None else context_class
        context_model_name_or_path = question_model_name_or_path if context_model_name_or_path is None else context_model_name_or_path
        
        # init encoders
        self.question_model = get_pretrained(question_class, pretrained_model_name_or_path=question_model_name_or_path)
        if context_class == 'shared':
            assert context_model_name_or_path == question_model_name_or_path
            self.context_model = self.question_model
            print(f"Sharing {self.question_model.__class__.__name__} to encode both questions and passages")
            self.shared_encoders = True
        else:
            self.shared_encoders = False
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
        
        global_question_embeddings = self.all_gather(outputs.question_pooler_output, sync_grads=True)
        global_context_embeddings = self.all_gather(outputs.context_pooler_output, sync_grads=True)
        global_labels = self.all_gather(local_labels, sync_grads=True)
        
        # reshape after all_gather
        if global_question_embeddings.ndim > 2:
            n_gpus, N, _ = global_question_embeddings.shape
            global_question_embeddings = global_question_embeddings.reshape(n_gpus*N, -1)
            _, N_times_M, _ = global_context_embeddings.shape
            global_context_embeddings = global_context_embeddings.reshape(n_gpus*N_times_M, -1)
            # labels are defined at the batch-level so we need to shift them when concatening batches
            for i in range(1, n_gpus):
                not_masked = (global_labels[i] != self.loss_fct.ignore_index)
                global_labels[i, not_masked] += i*N_times_M
            global_labels = global_labels.reshape(n_gpus*N)

        # compute similarity
        similarities = global_question_embeddings @ global_context_embeddings.T  # (N, N*M)
        log_probs = self.log_softmax(similarities)

        loss = self.loss_fct(log_probs, global_labels)
        return dict(loss=loss, log_probs=log_probs, labels=global_labels)   
    
    def eval_epoch_end(self, eval_outputs):
        return {'metrics': retrieval(eval_outputs, ignore_index=self.loss_fct.ignore_index)}
    
    def save_pretrained(self, ckpt_path, bert=False):
        question_model = self.question_model
        if self.shared_encoders:
            if bert:
                question_model = _get_bert(question_model)
            question_model.save_pretrained(ckpt_path)
        else:
            context_model = self.context_model
            if bert:
                question_model = _get_bert(question_model)
                context_model = _get_bert(context_model)
                question_path = ckpt_path/'question_model_bert'
                context_path = ckpt_path/'context_model_bert'
            else:
                question_path = ckpt_path/'question_model'
                context_path = ckpt_path/'context_model'
            question_model.save_pretrained(question_path)
            context_model.save_pretrained(context_path)
        
  
# TODO override load_from_checkpoint to use from_pretrained instead ?   
# see lightning/src/pytorch_lightning/core/saving.py
class ReRanker(Trainee):
    """    
    Parameters
    ----------
    model_kwargs: dict[str, str]
        Passed to get_pretrained
    metric_kwargs: dict[str, str], optional
        Passed to ranx.evaluate to compute metrics during evaluation
    """
    def __init__(self, *args, model_kwargs, metric_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_pretrained(**model_kwargs)
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean')
        ks = metric_kwargs.pop("ks", [1, 5, 10, 20, 100])
        default_metrics_kwargs = dict(metrics=[f"{m}@{k}" for m in ["mrr", "precision", "hit_rate"] for k in ks])
        default_metrics_kwargs.update(metric_kwargs)
        self.metrics_kwargs = default_metrics_kwargs
        self.post_init()   

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def step(self, inputs, _):   
        labels = inputs.pop('labels', None)
        question_ids = inputs.pop('ids', None)
        outputs = self(**inputs)
        M = self.trainer.datamodule.M
        n_times_m, _ = outputs.logits.shape
        assert n_times_m % M == 0
        N = n_times_m//M
        logits = outputs.logits.reshape(N, M)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None
        return dict(loss=loss, logits=logits, labels=labels, ids=question_ids)        
        
    def eval_epoch_end(self, eval_outputs):
        # rerank results of IR
        if isinstance(self.trainer.datamodule, ReRankerDataModule):
            run = get_run(eval_outputs, ir_run=self.trainer.datamodule.run)
            metrics = ranx.evaluate(qrels=self.trainer.datamodule.qrels, run=run, **self.metrics_kwargs)
        # in-batch metrics (e.g. for ICT)
        else:
            run = None
            metrics = retrieval(eval_outputs, ignore_index=self.loss_fct.ignore_index, output_key='logits')
        return {'metrics': metrics, 'run': run}
    
    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end, log and save run"""
        outputs = self.eval_epoch_end(*args, **kwargs)
        print(to_latex(outputs['metrics']))
        log_dir = Path(self.trainer.log_dir)
        for k, v in outputs['metrics'].items():
            self.log(f"test/{k}", v)
        with open(log_dir/'metrics.json', 'wt') as file:
            json.dump(outputs['metrics'], file)
        if outputs['run'] is not None:
            outputs['run'].save(log_dir/'run.json')
    
    def save_pretrained(self, ckpt_path, bert=False):
        assert not bert
        self.model.save_pretrained(ckpt_path)


def power_range(maximum):
    i = 0
    while True:
        p = min(2**i, maximum)
        yield p
        if p >= maximum:
            break
        i += 1
        
    
class Reader(Trainee):
    """    
    Parameters
    ----------
    model_kwargs: dict[str, str]
        Passed to get_pretrained
    tune_M: bool, optional
        Instead of extracting answers from the top-M input passages, 
        try every value in {2^i, s.t. 2^i <= M}
        Defaults to False (use only self.trainer.datamodule.M)
    """
    def __init__(self, *args, model_kwargs, tune_M=False, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO refactor all get_pretrained calls like this, 
        # no need to have a bunch of '*_name_or_path' in each Trainee model
        self.model = get_pretrained(**model_kwargs)
        self.tune_M = tune_M
        
        self.post_init()   
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def step(self, inputs, _):   
        keep_for_eval = {k: inputs.pop(k, None) for k in ['answer_strings', 'passage_scores']}
        model_outputs = self(**inputs)
        if 'text_inputs' in inputs:
            keep_for_eval['input_ids'] = inputs['text_inputs']['input_ids']
        else:
            keep_for_eval['input_ids'] = inputs['input_ids']
        keep_for_eval.update(model_outputs)
        return keep_for_eval
    
    def log_probs_to_answers(self, start_log_probs, end_log_probs, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        # TODO pass batch size
        passage_indices, start_indices, end_indices = batched_get_best_spans(
            start_probs=np.exp(start_log_probs),
            end_probs=np.exp(end_log_probs),
            **kwargs
        )
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.trainer.datamodule.tokenizer.batch_decode(answers, skip_special_tokens=True)
    
    def eval_epoch_end(self, eval_outputs):        
        M = self.trainer.datamodule.M
        # gather all outputs
        all_input_ids, all_start_log_probs, all_end_log_probs, all_passage_scores, all_answer_strings = [], [], [], [], []
        dataset_size = 0
        for batch in eval_outputs:
            input_ids = batch['input_ids']
            n_times_m, L = input_ids.shape
            assert n_times_m % M == 0
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
            
            all_input_ids.append(input_ids)
            all_start_log_probs.append(start_log_probs)
            all_end_log_probs.append(end_log_probs)
            passage_scores = batch['passage_scores']
            if passage_scores is not None:
                passage_scores = passage_scores.detach().cpu().numpy().reshape(N, M)
                all_passage_scores.append(passage_scores)
        assert len(all_answer_strings) == dataset_size
        # concat gathered outputs
        all_input_ids = pad_and_cat(all_input_ids)
        all_start_log_probs = pad_and_cat(all_start_log_probs)
        all_end_log_probs = pad_and_cat(all_end_log_probs)        
        if all_passage_scores:
            all_passage_scores = np.concatenate(all_passage_scores, axis=0)
        else:
            all_passage_scores = None
        if self.tune_M:
            return self.M_tuning(all_start_log_probs, all_end_log_probs, all_input_ids, all_answer_strings, all_passage_scores)   
        predictions = self.log_probs_to_answers(all_start_log_probs, all_end_log_probs, all_input_ids)
         # compute metrics        
        metrics = squad(predictions=predictions, references=all_answer_strings)
        if all_passage_scores is not None:
            weighted_predictions = self.log_probs_to_answers(all_start_log_probs, all_end_log_probs, all_input_ids, weights=all_passage_scores)
            weighted_metrics = squad(predictions=weighted_predictions, references=all_answer_strings)
            for k, v in weighted_metrics.items():
                 metrics['weighted_'+k] = v
        return {'metrics': metrics}
     
    def M_tuning(self, all_start_log_probs, all_end_log_probs, all_input_ids, all_answer_strings, all_passage_scores=None):
        N, M, L = all_input_ids.shape
        metrics_wrt_m = []
        for m in tqdm(list(power_range(M))):
            input_ids = all_input_ids[:, :m]
            start_log_probs = all_start_log_probs[:, :m]
            end_log_probs = all_end_log_probs[:, :m]
            predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids)
            metrics = squad(predictions=predictions, references=all_answer_strings)
            metrics['@M'] = m
            if all_passage_scores is not None:
                passage_scores = all_passage_scores[:, :m]
                weighted_predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids, weights=passage_scores)
                weighted_metrics = squad(predictions=weighted_predictions, references=all_answer_strings)
                for k, v in weighted_metrics.items():
                    metrics['weighted_'+k] = v
            metrics_wrt_m.append(metrics)
        with open(Path(self.trainer.log_dir)/'metrics_wrt_m.json', 'wt') as file:
            json.dump(metrics_wrt_m, file)
        # return metric with the best F1 score for logging
        # TODO option for what is best
        best = max(metrics_wrt_m, key=lambda metrics: metrics['f1'])
        return {'metrics': best}
    
    def save_pretrained(self, ckpt_path, bert=False):
        assert not bert
        self.model.save_pretrained(ckpt_path)
