"""
Trainee is a pl.LightningModule that computes the loss so it is compatible with Trainer.
"""
import warnings
from functools import partial
import re
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional

import ranx

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from ..data.loading import get_pretrained
from ..data.utils import to_latex
from ..models.qa import get_best_spans
from ..models.outputs import BiEncoderOutput, JointBiEncoderAndClipOutput, JointMonoAndCrossModalOutput
from .optim import LinearLRWithWarmup
from .metrics import batch_retrieval, squad_per_question, get_run, accumulate_batch_metrics, retrieval
from .data import ReRankerDataModule


def batched_cpu(batch):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


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
    output_cpu: bool, optional
    """
    def __init__(self, *args, freeze_regex=None, gradient_checkpointing=False,
                 warmup_steps=0, lr=2e-5, betas=(0.9, 0.999), eps=1e-08, 
                 weight_decay=0.0, output_cpu=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_regex = freeze_regex
        self.gradient_checkpointing = gradient_checkpointing
        self.weights_to_log = {}
        
        # scheduling and optimization
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay        
        self.param_groups = self.parameters()
        self.output_cpu = output_cpu
        
    # should be called at the end of each subclass __init__
    def post_init(self):
        if self.freeze_regex is not None:
            self.freeze(self.freeze_regex)        
        if self.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
    def step(self, batch, batch_idx):
        raise NotImplementedError("Subclass and implement step.")
    
    def eval_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
    
    def log(self, name, value, **kwargs):
        """Ignores None values."""
        if value is None:
            return None
        super().log(name, value, **kwargs)
            
    def training_step(self, batch, batch_idx):
        """Step and log training metrics"""
        outputs = self.step(batch, batch_idx)
        self.log("train/loss", outputs['loss'])
        for name, tensor in self.weights_to_log.items():
            self.log(f"weights/{name}", tensor.cpu().detach().item())

        return outputs
    
    def validation_step(self, batch, batch_idx):
        """Step and log validation metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("eval/loss", outputs['loss'])
        if self.output_cpu:
            return batched_cpu(outputs)
        return outputs
    
    def test_step(self, batch, batch_idx):
        """Step and log test metrics"""
        outputs = self.eval_step(batch, batch_idx)
        self.log("test/loss", outputs['loss'])
        if self.output_cpu:
            return batched_cpu(outputs)
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
        optimizer = AdamW(self.param_groups, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        
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
        self.post_init()   
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def step(self, batch, batch_idx):
        labels = batch.pop('labels', None)
        if labels is not None:
            warnings.warn(
                f"The loss is computed by {self.model.__class__.__name__}, "
                f"labels will have no effect."
            )
        outputs = self(**batch, return_loss=True, return_dict=True)
        return {'loss': outputs['loss'], 'logits_per_image': outputs['logits_per_image']}
    
    def eval_step(self, inputs, batch_idx):
        outputs = self.step(inputs, batch_idx)
        logits_per_image = outputs['logits_per_image']
        labels = torch.arange(logits_per_image.shape[1], dtype=torch.long)
        metrics = batch_retrieval(logits_per_image, labels)
        return {'loss': outputs['loss'], 'metrics': metrics}
    
    def eval_epoch_end(self, eval_outputs):
        metrics = accumulate_batch_metrics([output['metrics'] for output in eval_outputs])
        return {'metrics': metrics}
    
    def save_pretrained(self, ckpt_path, bert=False):
        assert not bert
        self.model.save_pretrained(ckpt_path)
        
        
class JointMonoAndCrossModal(Trainee):
    def __init__(self, *args, model_kwargs: dict, image_weight=0.5, cm_weight=0.5, 
                 learn_weights=False, mm_weights_lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_pretrained(**model_kwargs)
        self.log_softmax = nn.LogSoftmax(1)
        self.loss_fct = nn.NLLLoss(reduction='mean')
        self.image_weight = nn.Parameter(
            torch.tensor([image_weight]), 
            requires_grad=learn_weights
        )
        self.cm_weight = nn.Parameter(
            torch.tensor([cm_weight]), 
            requires_grad=learn_weights
        )
        self.weights_to_log.update({
            "image_weight": self.image_weight,
            "cm_weight": self.cm_weight,
            "temperature": self.model.logit_scale,
        })
            
        self.param_groups = [
            {'params': self.model.parameters()},            
            {
                'params': [self.image_weight, self.cm_weight],
                'lr': mm_weights_lr if mm_weights_lr is not None else self.lr
            },
        ]
        
        self.post_init()   
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        paired_pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ):       
        question_images = self.model.vision_model(pixel_values)
        question_images = self.model.visual_projection(question_images[1])
        question_images = question_images / question_images.norm(p=2, dim=-1, keepdim=True)
        
        context_titles = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        context_titles = self.model.text_projection(context_titles[1])
        context_titles = context_titles / context_titles.norm(p=2, dim=-1, keepdim=True)
        
        context_images = self.model.vision_model(paired_pixel_values)
        context_images = self.model.visual_projection(context_images[1])
        context_images = context_images / context_images.norm(p=2, dim=-1, keepdim=True)
            
        return JointMonoAndCrossModalOutput(
            question_images=question_images,
            context_images=context_images,
            context_titles=context_titles
        )
        
    def step(self, inputs, _):       
        # TODO multi-gpu
        local_labels = inputs.pop('labels')
        outputs = self(**inputs)
                
        image_similarities = self.model.logit_scale.exp() * (outputs.question_images @ outputs.context_images.T)
        cm_similarities = self.model.logit_scale.exp() * (outputs.question_images @ outputs.context_titles.T)
        similarities = self.image_weight*image_similarities + self.cm_weight*cm_similarities
        
        # note that this loss is asymmetrical unlike CLIP implemented in CrossModal
        log_probs = self.log_softmax(similarities)
        loss = self.loss_fct(log_probs, local_labels)
        
        return dict(loss=loss, log_probs=log_probs, local_labels=local_labels,
                    image_similarities=image_similarities,
                    cm_similarities=cm_similarities)
        
    def eval_step(self, inputs, batch_idx):
        model_outputs = self.step(inputs, batch_idx)
        local_labels = model_outputs['local_labels']
        metrics = batch_retrieval(model_outputs['log_probs'], local_labels, ignore_index=self.loss_fct.ignore_index)
        image_metrics = batch_retrieval(model_outputs['image_similarities'], local_labels, ignore_index=self.loss_fct.ignore_index)
        cm_metrics = batch_retrieval(model_outputs['cm_similarities'], local_labels, ignore_index=self.loss_fct.ignore_index)
        
        return dict(
            loss=model_outputs['loss'], metrics=metrics, image_metrics=image_metrics, cm_metrics=cm_metrics
        )   
        
    def eval_epoch_end(self, eval_outputs):
        metrics = accumulate_batch_metrics([output['metrics'] for output in eval_outputs])
        for model in ['image', 'cm']:
            model_metrics = accumulate_batch_metrics([output[f"{model}_metrics"] for output in eval_outputs])
            metrics.update({f"{model}_{k}": v for k, v in model_metrics.items()})
        return {'metrics': metrics}
    
    def save_pretrained(self, ckpt_path, bert=False):
        assert not bert
        self.model.save_pretrained(ckpt_path)
        mm_weights = {                
            "image_weight": (self.image_weight * self.model.logit_scale.exp()).item(),
            "cm_weight": (self.cm_weight * self.model.logit_scale.exp()).item()
        }
        with open(ckpt_path/'mm_weights.json','wt') as file:
            json.dump(mm_weights, file)
            
        
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
    question_kwargs, context_kwargs: dict, optional
    superclass: bool, optional
        Means that BiEncoder is instantiated from a subclass. Disables post_init.
        Defaults to False.
    """
    def __init__(self, *args, question_class, question_model_name_or_path, 
                 context_class=None, context_model_name_or_path=None, 
                 question_kwargs={}, context_kwargs=None, superclass=False, **kwargs):
        super().__init__(*args, **kwargs)        
        # default to symmetric encoders
        context_class = question_class if context_class is None else context_class
        context_model_name_or_path = question_model_name_or_path if context_model_name_or_path is None else context_model_name_or_path
        context_kwargs = question_kwargs.copy() if context_kwargs is None else context_kwargs
        # init encoders
        self.question_model = get_pretrained(question_class, pretrained_model_name_or_path=question_model_name_or_path, **question_kwargs)
        if context_class == 'shared':
            assert context_model_name_or_path == question_model_name_or_path
            self.context_model = self.question_model
            print(f"Sharing {self.question_model.__class__.__name__} to encode both questions and passages")
            self.shared_encoders = True
            self.weights_to_log.update(getattr(self.question_model, 'weights_to_log', {}))
        else:
            self.shared_encoders = False
            self.context_model = get_pretrained(context_class, pretrained_model_name_or_path=context_model_name_or_path, **context_kwargs)
            for name, weight in getattr(self.question_model, 'weights_to_log', {}).items():
                self.weights_to_log[f"question_{name}"] = weight
            for name, weight in getattr(self.context_model, 'weights_to_log', {}).items():
                self.weights_to_log[f"context_{name}"] = weight
        
        # loss and metrics
        self.log_softmax = nn.LogSoftmax(1)
        self.loss_fct = nn.NLLLoss(reduction='mean')
        
        if not superclass:
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
        return dict(loss=loss, log_probs=log_probs, global_labels=global_labels)
                    
    def eval_step(self, inputs, batch_idx):
        model_outputs = self.step(inputs, batch_idx)
        metrics = batch_retrieval(model_outputs['log_probs'], model_outputs['global_labels'], ignore_index=self.loss_fct.ignore_index)
        return dict(loss=model_outputs['loss'], metrics=metrics)   
                
    def eval_epoch_end(self, eval_outputs):
        metrics = accumulate_batch_metrics([output['metrics'] for output in eval_outputs])
        return {'metrics': metrics}
    
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
    
    
class JointBiEncoderAndClip(BiEncoder):
    def __init__(self, *args, clip, question_weight=1/3, image_weight=1/3, cm_weight=1/3, 
                 learn_weights=False, clip_lr=None, mm_weights_lr=None, **kwargs):        
        super().__init__(*args, superclass=True, **kwargs)      
        self.clip = get_pretrained(**clip)
        self.question_weight = nn.Parameter(
            torch.tensor([question_weight]), 
            requires_grad=learn_weights
        )
        self.image_weight = nn.Parameter(
            torch.tensor([image_weight]), 
            requires_grad=learn_weights
        )
        self.cm_weight = nn.Parameter(
            torch.tensor([cm_weight]), 
            requires_grad=learn_weights
        )
        self.weights_to_log.update({
            "question_weight": self.question_weight,
            "image_weight": self.image_weight,
            "cm_weight": self.cm_weight,
            "temperature": self.clip.logit_scale,
        })
            
        self.param_groups = [
            {'params': self.question_model.parameters()},
            {'params': self.context_model.parameters()},
            {
                'params': self.clip.parameters(),
                'lr': clip_lr if clip_lr is not None else self.lr
            },            
            {
                'params': [self.question_weight, self.image_weight, self.cm_weight],
                'lr': mm_weights_lr if mm_weights_lr is not None else self.lr
            },
        ]
        
        self.post_init()        
        
    def forward(self, question_inputs, context_inputs):
        # TODO do not compute for modules with weight=0
        
        # embed question-image and context_image
        question_images = self.clip.get_image_features(
            question_inputs.pop('pixel_values'), 
            return_dict=False
        )
        question_images = question_images / question_images.norm(p=2, dim=-1, keepdim=True)
        context_images = self.clip.get_image_features(
            context_inputs.pop('pixel_values'),
            return_dict=False
        )
        context_images = context_images / context_images.norm(p=2, dim=-1, keepdim=True)
        
        # embed context titles
        context_titles = self.clip.get_text_features(
            **context_inputs.pop('titles'), 
            return_dict=False
        )
        context_titles = context_titles / context_titles.norm(p=2, dim=-1, keepdim=True)
        
        # embed questions and contexts
        question_outputs = self.question_model(**question_inputs)
        context_outputs = self.context_model(**context_inputs)

        return JointBiEncoderAndClipOutput(
            question_pooler_output=question_outputs.pooler_output,
            context_pooler_output=context_outputs.pooler_output,
            question_images=question_images,
            context_images=context_images,
            context_titles=context_titles
        )
        
    def step(self, inputs, _):       
        # TODO multi-gpu
        local_labels = inputs.pop('labels')  # (N, )
        outputs = self(**inputs)
                
        question_similarities = self.question_weight * (outputs.question_pooler_output @ outputs.context_pooler_output.T)  # (N, N*M)
        image_similarities = self.image_weight * self.clip.logit_scale.exp() * (outputs.question_images @ outputs.context_images.T)
        cm_similarities = self.cm_weight * self.clip.logit_scale.exp() * (outputs.question_images @ outputs.context_titles.T)
        similarities = question_similarities + image_similarities + cm_similarities
        
        log_probs = self.log_softmax(similarities)
        loss = self.loss_fct(log_probs, local_labels)
        
        return dict(loss=loss, log_probs=log_probs, local_labels=local_labels,
                    question_similarities=question_similarities, 
                    image_similarities=image_similarities,
                    cm_similarities=cm_similarities)
        
    def eval_step(self, inputs, batch_idx):
        model_outputs = self.step(inputs, batch_idx)
        local_labels = model_outputs['local_labels']
        metrics = batch_retrieval(model_outputs['log_probs'], local_labels, ignore_index=self.loss_fct.ignore_index)
        question_metrics = batch_retrieval(model_outputs['question_similarities'], local_labels, ignore_index=self.loss_fct.ignore_index)
        image_metrics = batch_retrieval(model_outputs['image_similarities'], local_labels, ignore_index=self.loss_fct.ignore_index)
        cm_metrics = batch_retrieval(model_outputs['cm_similarities'], local_labels, ignore_index=self.loss_fct.ignore_index)
        
        return dict(
            loss=model_outputs['loss'], metrics=metrics, question_metrics=question_metrics,
            image_metrics=image_metrics, cm_metrics=cm_metrics
        )   
        
    def eval_epoch_end(self, eval_outputs):
        metrics = accumulate_batch_metrics([output['metrics'] for output in eval_outputs])
        for model in ['question', 'image', 'cm']:
            model_metrics = accumulate_batch_metrics([output[f"{model}_metrics"] for output in eval_outputs])
            metrics.update({f"{model}_{k}": v for k, v in model_metrics.items()})
        return {'metrics': metrics}
    
    def save_pretrained(self, ckpt_path, bert=False):            
        self.clip.save_pretrained(ckpt_path/'clip')
        question_model = self.question_model
        if self.shared_encoders:
            if bert:
                question_model = _get_bert(question_model)
            question_model.save_pretrained(ckpt_path/'bert')
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
        mm_weights = {                
            "question_weight": self.question_weight.item(),
            "image_weight": (self.image_weight * self.clip.logit_scale.exp()).item(),
            "cm_weight": (self.cm_weight * self.clip.logit_scale.exp()).item()
        }
        with open(ckpt_path/'mm_weights.json','wt') as file:
            json.dump(mm_weights, file)
    
    
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
        super().__init__(*args, output_cpu=True, **kwargs)
        self.model = get_pretrained(**model_kwargs)
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean')
        ks = metric_kwargs.pop("ks", [1, 5, 10, 20, 100])
        default_metrics_kwargs = dict(metrics=[f"{m}@{k}" for m in ["mrr", "precision", "hit_rate"] for k in ks])
        default_metrics_kwargs.update(metric_kwargs)
        self.metrics_kwargs = default_metrics_kwargs
        self.weights_to_log.update(getattr(self.model, 'weights_to_log', {}))
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
        self.weights_to_log.update(getattr(self.model, 'weights_to_log', {}))
        self.tune_M = tune_M
        
        self.post_init()   
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def step(self, inputs, _):             
        answer_strings = inputs.pop('answer_strings', None)
        if not self.model.fuse_ir_score:
            passage_scores = inputs.pop('passage_scores', None)
        else:
            passage_scores = None    
        model_outputs = self(**inputs)
        model_outputs['answer_strings'] = answer_strings
        model_outputs['passage_scores'] = passage_scores
        return model_outputs
    
    def eval_step(self, inputs, batch_idx):    
        model_outputs = self.step(inputs, batch_idx)
        answer_strings = model_outputs['answer_strings']
        passage_scores = model_outputs['passage_scores']
        if 'text_inputs' in inputs:
            input_ids = inputs['text_inputs']['input_ids']
        else:
            input_ids = inputs['input_ids']
            
        if self.tune_M:            
            raise NotImplementedError()
        
        # compute metrics      
        M = self.trainer.datamodule.M
        n_times_m, L = input_ids.shape
        assert n_times_m % M == 0
        N = n_times_m//M
        answer_strings = [answer_strings[i] for i in range(0, len(answer_strings), M)]
        assert len(answer_strings) == N
        input_ids = input_ids.reshape(N, M, L)  
        start_log_probs = model_outputs['start_log_probs'].reshape(N, M, L)
        end_log_probs = model_outputs['end_log_probs'].reshape(N, M, L)
        predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids)
        
        metrics = squad_per_question(predictions=predictions, references=answer_strings)
        
        if passage_scores is not None:
            weighted_predictions = self.log_probs_to_answers(start_log_probs, end_log_probs, input_ids, weights=passage_scores)
            weighted_metrics = squad_per_question(predictions=weighted_predictions, references=answer_strings)
        else:
            weighted_predictions = None
            weighted_metrics = None
        return {'loss': model_outputs['loss'], 'metrics': metrics, 'predictions': predictions, 
                'weighted_metrics': weighted_metrics, 'weighted_predictions': weighted_predictions}
    
    def log_probs_to_answers(self, start_log_probs, end_log_probs, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        passage_indices, start_indices, end_indices = get_best_spans(
            start_probs=start_log_probs.exp(),
            end_probs=end_log_probs.exp(),
            **kwargs
        )
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.trainer.datamodule.tokenizer.batch_decode(answers, skip_special_tokens=True)
    
    def eval_epoch_end(self, eval_outputs):     
        metrics = {"exact_match": [], "f1": [], "weighted_exact_match": [], "weighted_f1": []}
        predictions, weighted_predictions = [], []
        for eval_output in eval_outputs:
            for k, v in eval_output['metrics'].items():
                metrics[k].extend(v)
            predictions.extend(eval_output['predictions'])
            if eval_output['weighted_metrics'] is not None:
                for k, v in eval_output['weighted_metrics'].items():
                    metrics["weighted_"+k].extend(v)
                weighted_predictions.extend(eval_output['weighted_predictions'])
        for k, v in metrics.items():
            if v:
                metrics[k] = sum(v)/len(v)
            else:
                metrics[k] = None
        return {'metrics': metrics, 'predictions': predictions, 'weighted_predictions': weighted_predictions}
    
    def test_epoch_end(self, *args, **kwargs):
        """eval_epoch_end and log"""
        eval_output = self.eval_epoch_end(*args, **kwargs)
        metrics = eval_output['metrics']
        print(to_latex(metrics))
        log_dir = Path(self.trainer.log_dir)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        with open(log_dir/'metrics.json', 'wt') as file:
            json.dump(metrics, file)
        with open(log_dir/'predictions.json', 'wt') as file:
            json.dump(eval_output['predictions'], file)
        if eval_output['weighted_predictions']:
            with open(log_dir/'weighted_predictions.json', 'wt') as file:
                json.dump(eval_output['weighted_predictions'], file)
            
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
