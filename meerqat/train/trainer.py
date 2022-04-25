"""Usage: trainer.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import warnings
from tqdm import tqdm
import collections
import sys
import logging
import humanize
import re

import numpy as np
import torch
from torch import nn
from torch.autograd import set_detect_anomaly
from torch.utils.data.dataset import IterableDataset
import torch.distributed as dist

from transformers import Trainer, TrainingArguments, trainer_callback, logging as t_logging
from transformers.trainer_callback import TrainerState
from datasets import load_from_disk, load_metric
from transformers.deepspeed import deepspeed_init
from transformers.file_utils import WEIGHTS_NAME, is_torch_tpu_available
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify
)
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
if is_torch_tpu_available():
    import torch_xla.distributed.parallel_loader as pl

from meerqat.data.loading import load_pretrained_in_kwargs
from meerqat.models.qa import get_best_spans, format_predictions_for_squad
from meerqat.models.utils import debug_shape
from meerqat.train import metrics as metric_functions


logging.basicConfig()
logger = logging.getLogger(__name__)


def max_memory_usage(human=False):
    logs = {}
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        value = torch.cuda.max_memory_allocated(device)
        if human:
            value = humanize.naturalsize(value, gnu=True)
        logs[f"max_memory_{device}"] = value
    return logs


class MeerqatTrainer(Trainer):
    """Base class for all trainers. Should be very similar to Trainer"""
    def __init__(self, model, *args, freeze=None, **kwargs):
        if freeze is not None:
            model = self.freeze(model, freeze)
        super().__init__(model, *args, **kwargs)
        self.prediction_file_name = "predictions.json"
        self.metrics_file_name = "metrics.json"
    
    def freeze(self, model, regex):
        regex = re.compile(regex)
        total, frozen = 0, 0
        logger.debug("Model parameters:\t\t\t\tName\t#Trainable\t#Total")
        for name, param in model.named_parameters():
            numel = param.numel()
            if regex.match(name):
                param.requires_grad = False
                frozen += numel
            total += numel
            logger.debug(f"{name}\t\t{(numel if param.requires_grad else 0):,d}\t{numel:,d}")
        logger.info(f"Froze {frozen:,d} parameters out of {total:,d}")
        return model
        
    def log(self, logs: dict) -> None:
        """Adds memory usage to the logs"""
        logs.update(max_memory_usage())
        return super().log(logs)

    def write_predictions(self, predictions, resume_from_checkpoint):
        if isinstance(predictions, (list, dict)):
            with open(resume_from_checkpoint/self.prediction_file_name, "w") as file:
                json.dump(predictions, file)
        else:
            raise NotImplementedError()

    def write_metrics(self, metrics, resume_from_checkpoint):
        print(metrics)
        with open(resume_from_checkpoint/self.metrics_file_name, "w") as file:
            json.dump(metrics, file)


class QuestionAnsweringTrainer(MeerqatTrainer):
    """
    Base class for Question Answering trainers. Should work for both IR and RC.

        Overrides some methods because we need to create the batch of questions and passages on-the-fly

    Because the inputs should be shaped like (N * M, L), where:
            N - number of distinct questions
            M - number of passages per question in a batch
            L - sequence length

    Parameters
    ----------
    *args, **kwargs: additional arguments are passed to MeerqatTrainer
    kb: str, optional
        path towards the knowledge base (Dataset) used to get the passages
        Optional because not needed in ICTTrainer, mandatory for the other trainers.
    M: int, optional
        Number of passages (relevant or irrelevant) per question in a batch
        Defaults to 24
    n_relevant_passages: int, optional
        Defaults to 1
    search_key: str, optional
        This column in the dataset suffixed by '_indices' and '_scores' should hold the result of information retrieval
        used during evaluation (e.g. the output of ir.search)
        Suffixed by "_provenance_indices" and "_irrelevant_indices" it should hold:
            1. the union of relevant search and provenance_indices
            2. irrelevant results from the search
        used during training (according to M and n_relevant_passages)
        Defaults to 'search'
    tokenization_kwargs: dict, optional
        To be passed to self.tokenizer
    """
    def __init__(self, *args, kb=None, M=24, n_relevant_passages=1, search_key='search', tokenization_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer is not None
        if kb is not None:
            self.kb = load_from_disk(kb)
        else:
            self.kb = None
        self.M = M
        assert n_relevant_passages <= M
        self.n_relevant_passages = n_relevant_passages
        self.search_key = search_key
        default_tokenization_kwargs = dict(return_tensors='pt', padding='max_length', truncation=True)
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        default_tokenization_kwargs.update(tokenization_kwargs)
        self.tokenization_kwargs = default_tokenization_kwargs
        self.data_collator = self.collate_fn

        # we need those ‘un-used’ columns to actually create the batch the model will use
        if self.args.remove_unused_columns:
            warnings.warn(f'Setting args.remove_unused_columns to False')
            self.args.remove_unused_columns = False

    def get_training_passages(self, item):
        relevant_passages = []
        all_relevant_indices = item[self.search_key+"_provenance_indices"]
        n_relevant = min(len(all_relevant_indices), self.n_relevant_passages)
        if n_relevant > 0:
            relevant_indices = np.random.choice(all_relevant_indices, n_relevant, replace=False)
            if len(relevant_indices) > 0:
                relevant_passages = self.kb.select(relevant_indices)
        irrelevant_passages = []
        all_irrelevant_indices = item[self.search_key+"_irrelevant_indices"]
        n_irrelevant = min(len(all_irrelevant_indices), self.M-self.n_relevant_passages)
        if n_irrelevant > 0:
            irrelevant_indices = np.random.choice(all_irrelevant_indices, n_irrelevant, replace=False)
            if len(irrelevant_indices) > 0:
                irrelevant_passages = self.kb.select(irrelevant_indices)
        elif n_relevant <= 0:
            warnings.warn(f"Didn't find any passage for question {item['id']}")
        return relevant_passages, irrelevant_passages


class DPRBiEncoderTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_softmax = nn.LogSoftmax(1)
        self.loss_fct = nn.NLLLoss(reduction='mean')
        assert self.n_relevant_passages == 1

    def collate_fn(self, items):
        """
        Collate batch so that each question is associate with n_relevant_passages and M-n irrelevant ones.
        Also tokenizes input strings

        N - number of questions in a batch
        M - number of passages per questions
        d - dimension of the model/embeddings

        Returns (a dict of)
        -------------------
        question_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N, L)
            **kwargs: more tensors depending on the tokenizer, e.g. attention_mask
        context_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N*M, L)
                The first N rows correspond to the relevant contexts for the N questions
                The rest N*(M-1) rows are irrelevant contexts for all questions.
            **kwargs: idem
        """
        n_irrelevant_passages = self.M-self.n_relevant_passages
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        for i, item in enumerate(items):
            relevant_passage, irrelevant_passage = self.get_training_passages(item)
            relevant_passage, irrelevant_passage = relevant_passage['passage'], irrelevant_passage['passage']
            if len(relevant_passage) < 1:
                relevant_passage = ['']
                labels.append(self.loss_fct.ignore_index)
            else:
                labels.append(i)
            if len(irrelevant_passage) < n_irrelevant_passages:
                irrelevant_passage.extend(['']*(n_irrelevant_passages-len(irrelevant_passage)))
            questions.append(item['input'])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)

        question_inputs = self.tokenizer(questions, **self.tokenization_kwargs)
        context_inputs = self.tokenizer(relevant_passages + irrelevant_passages, **self.tokenization_kwargs)
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations across all the nodes.
        Adapted from https://github.com/facebookresearch/DPR/blob/main/train_dense_encoder.py

        N. B. this means that the whole representations of questions and contexts, and their similarity matrix, must fit on a single GPU.
        """
        if self.label_smoother is not None:
            raise NotImplementedError()

        local_labels = inputs.pop('labels', None)  # (N, )

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if local_labels is None:
            # FIXME: this returns representations and not similarities
            return (None, outputs) if return_outputs else None

        local_question_representations = outputs.question_pooler_output  # (N, d)
        local_context_representations = outputs.context_pooler_output  # (N*M, d)
        if self.args.world_size > 1:
            # copies local representations (in DPR they are moved to CPU but I got a RuntimeError: "Tensors must be CUDA")
            question_representations_to_send = torch.empty_like(local_question_representations).copy_(local_question_representations).detach_()
            context_representations_to_send = torch.empty_like(local_context_representations).copy_(local_context_representations).detach_()
            labels_to_send = torch.empty_like(local_labels).copy_(local_labels)

            # gathers representations from other GPUs
            question_representations_gatherer = [torch.empty_like(question_representations_to_send) for _ in range(self.args.world_size)]
            context_representations_gatherer = [torch.empty_like(context_representations_to_send) for _ in range(self.args.world_size)]
            labels_gatherer = [torch.empty_like(labels_to_send) for _ in range(self.args.world_size)]
            dist.all_gather(question_representations_gatherer, question_representations_to_send)
            dist.all_gather(context_representations_gatherer, context_representations_to_send)
            dist.all_gather(labels_gatherer, labels_to_send)
            
            # keep local vector in the local_rank index (taken from DPR, to not loose the gradients?)
            label_shift = 0
            global_question_representations, global_context_representations, global_labels = [], [], []
            gatherers = zip(question_representations_gatherer, context_representations_gatherer, labels_gatherer)
            for i, (received_question_representations, received_context_representations, received_labels) in enumerate(gatherers):
                # receiving representations from other GPUs
                if i != self.args.local_rank:
                    global_question_representations.append(received_question_representations.to(local_question_representations.device))
                    global_context_representations.append(received_context_representations.to(local_context_representations.device))
                    # labels are defined at the batch-level so we need to shift them when concatening batches
                    received_labels[received_labels!=self.loss_fct.ignore_index] += label_shift
                    label_shift += received_context_representations.shape[0]  # N*M
                    global_labels.append(received_labels.to(local_labels.device))
                # keep local representation
                else:
                    global_question_representations.append(local_question_representations)
                    global_context_representations.append(local_context_representations)
                    # labels are defined at the batch-level so we need to shift them when concatening batches
                    local_labels[local_labels!=self.loss_fct.ignore_index] += label_shift
                    label_shift += local_context_representations.shape[0]  # N*M
                    global_labels.append(local_labels)
            global_question_representations = torch.cat(global_question_representations, dim=0)
            global_context_representations = torch.cat(global_context_representations, dim=0)
            global_labels = torch.cat(global_labels, dim=0)
        else:
            global_question_representations = local_question_representations  # (N, d)
            global_context_representations = local_context_representations  # (N*M, d)
            global_labels = local_labels  # (N, )

        # compute similarity
        similarities = global_question_representations @ global_context_representations.T  # (N, N*M)
        log_probs = self.log_softmax(similarities)

        loss = self.loss_fct(log_probs, global_labels)

        # beware of https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L2513 !!
        # do NOT return log_probs outside of a dict else it will get truncated
        return (loss, dict(log_probs=log_probs)) if return_outputs else loss


class ILFTrainer(DPRBiEncoderTrainer):
    """
    Fuses DPR’s text representation with image embeddings by projecting them linearly in the same space
    --> loads pre-computed image features along with text 
    --> overrides collate_fn
    
    Parameters
    ----------
    image_kb: str, optional
        Path to the KB that holds pre-computed image features
        Can be mapped from kb using kb['index']
        Optional to ease inheritance
    """
    def __init__(self, *args, image_kb=None, **kwargs):
        super().__init__(*args, **kwargs)
        if image_kb is not None:
            self.image_kb = load_from_disk(image_kb)
        else:
            self.image_kb = None
        # image dimensions and model names
        self.n_faces = self.model.question_model.config.n_faces
        assert self.n_faces == self.model.context_model.config.n_faces
        
        assert(self.model.question_model.image_embeddings.keys() == self.model.context_model.image_embeddings.keys())  
        self.image_embeddings_keys = self.model.question_model.image_embeddings.keys()
        self.image_dims = {}
        for name in self.image_embeddings_keys:
            image_dim = self.model.question_model.image_embeddings[name].linear.in_features
            assert(image_dim == self.model.context_model.image_embeddings[name].linear.in_features)
            self.image_dims[name] = image_dim
            
        assert(self.model.question_model.face_embedding.face_proj.in_features == self.model.context_model.face_embedding.face_proj.in_features)
        self.face_dim = self.model.question_model.face_embedding.face_proj.in_features
        assert(self.model.question_model.face_embedding.bbox_proj.in_features == self.model.context_model.face_embedding.bbox_proj.in_features)
        self.bbox_dim = self.model.question_model.face_embedding.bbox_proj.in_features

    def get_face_inputs(self, items):
        # trim or pad, and convert to tensor
        face_embeddings = torch.zeros((len(items), self.n_faces, self.face_dim))
        face_boxes = torch.zeros((len(items), self.n_faces, self.bbox_dim))
        # 0=masked, 1=not masked
        attention_mask = torch.zeros((len(items), self.n_faces), dtype=torch.long)
        for i, item in enumerate(items):
            face_embedding = item.get("face_embedding")
            # can happen in two cases: 1. no face detected; 2. padding passage
            if face_embedding is None:
                # keep zero-padding/mask
                continue
            n_faces = min(self.n_faces, len(face_embedding))
            face_embeddings[i, : n_faces] = torch.tensor(face_embedding[: n_faces])
            bbox = item["face_box"]
            face_boxes[i, : n_faces] = torch.tensor(bbox[: n_faces])
            attention_mask[i, : n_faces] = 1

        face_inputs = {
            "face": face_embeddings,
            "bbox": face_boxes,
            "attention_mask": attention_mask
        }
        return face_inputs

    def get_image_inputs(self, items):
        image_inputs = {}
        for name in self.image_embeddings_keys: 
            features = torch.zeros(len(items), self.image_dims[name])
            # 0=masked, 1=not masked
            attention_mask = torch.zeros(len(items), dtype=torch.long)

            for i, item in enumerate(items):
                feature = item.get(name)
                # in case of padding passage
                if feature is None:
                    # keep zero-padding/mask
                    continue
                features[i] = torch.tensor(feature)
                attention_mask[i] = 1

            image_inputs[name] = dict(input=features, attention_mask=attention_mask)
        return image_inputs               
                                   
    def add_image_features(self, passages):
        output = []
        for passage in passages:
            image_item = self.image_kb[passage['index']]
            for k, v in image_item.items():
                passage.setdefault(k, v)
            output.append(passage)
        return output
        
    def collate_fn(self, items):
        # find relevant and irrelevant passages, pad if necessary
        n_irrelevant_passages = self.M-self.n_relevant_passages
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        for i, item in enumerate(items):
            relevant_passage, irrelevant_passage = self.get_training_passages(item)
            # Dataset to list (to get the same format as items)
            relevant_passage, irrelevant_passage = self.add_image_features(relevant_passage), self.add_image_features(irrelevant_passage)
            if len(relevant_passage) < 1:
                relevant_passage = [{'passage': ''}]
                labels.append(self.loss_fct.ignore_index)
            else:
                labels.append(i)
            if len(irrelevant_passage) < n_irrelevant_passages:
                irrelevant_passage.extend([{'passage': ''}]*(n_irrelevant_passages-len(irrelevant_passage)))
            questions.append(item['input'])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)

        # tokenize questions
        question_inputs_text = self.tokenizer(questions, **self.tokenization_kwargs)
        # concatenate passages and tokenize
        all_passages = relevant_passages + irrelevant_passages
        context_inputs_text = self.tokenizer([p['passage'] for p in all_passages], **self.tokenization_kwargs)

        # get image features, for both questions and passages
        question_inputs = dict(
            text_inputs=question_inputs_text, 
            face_inputs=self.get_face_inputs(items), 
            image_inputs=self.get_image_inputs(items)
        )
        context_inputs = dict(
            text_inputs=context_inputs_text, 
            face_inputs=self.get_face_inputs(all_passages), 
            image_inputs=self.get_image_inputs(all_passages)
        )

        # wrap it up
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch


class ICTTrainer(ILFTrainer):
    """
    Extends the Inverse Cloze Task (ICT, lee_latent_2019) to multimodal documents.
    Given a wikipedia section, one sentence is considered as a pseudo-question/query and the nearby sentences as a pseudo-target/relevant passage.
    In this multimodal setting, we also consider the image of the section in the query and the infobox/main image of the article in the target.

    Inherits from ILFTrainer/DPRBiEncoderTrainer and overrides:
    - get_training_passages, which implements what’s described above
    - collate_fn to load and concatenate the image features

    The image_kb, kb and search_key attributes are not used.

    References
    ----------
    @inproceedings{lee_latent_2019,
        address = {Florence, Italy},
        title = {Latent {Retrieval} for {Weakly} {Supervised} {Open} {Domain} {Question} {Answering}},
        url = {https://aclanthology.org/P19-1612},
        doi = {10.18653/v1/P19-1612},
        booktitle = {Proceedings of the 57th {Annual} {Meeting} of the {Association} for {Computational} {Linguistics}},
        publisher = {Association for Computational Linguistics},
        author = {Lee, Kenton and Chang, Ming-Wei and Toutanova, Kristina},
        month = jul,
        year = {2019},
        pages = {6086--6096}
    }
    """
    def __init__(self, *args, sentences_per_target=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.kb = None
        self.image_kb = None
        self.sentences_per_target = sentences_per_target
        self.data_collator = self.collate_fn

    def get_training_passages(self, item):
        """
        Beware this does not return the same data as the parent classes.

        Returns
        -------
        query: dict
        target: dict
        """
        sentences = item['sentences']

        # pick a random sentence: easy
        i = np.random.randint(len(sentences))
        query = dict(text=sentences[i]['text'])
        # pick n random sentences around it: more tricky
        n = min(self.sentences_per_target, len(sentences)-1)
        max_shift = min(i, n)
        if i+n < len(sentences):
            min_shift = 0
        else:
            min_shift = i + n - len(sentences) + 1
        shift = np.random.randint(min_shift, max_shift+1)
        target = [s['text'] for s in sentences[i-shift: i]+sentences[i+1: i+1+n-shift]]
        target = dict(text=" ".join(target))  

        # rename context image features
        for k in ({"face_box", "face_embedding"} | self.image_embeddings_keys):
            target[k] = item.get(f"context_{k}")
        return query, target

    def collate_fn(self, items):        
        questions, relevant_passages, labels = [], [], []
        for i, item in enumerate(items):
            query, relevant_passage = self.get_training_passages(item)
            labels.append(i)
            questions.append(query)
            relevant_passages.append(relevant_passage)

        question_inputs_text = self.tokenizer([q['text'] for q in questions], **self.tokenization_kwargs)
        context_inputs_text = self.tokenizer([p['text'] for p in relevant_passages], **self.tokenization_kwargs)
        # get image features, for both questions and passages
        question_inputs = dict(
            text_inputs=question_inputs_text, 
            face_inputs=self.get_face_inputs(items), 
            image_inputs=self.get_image_inputs(items)
        )
        context_inputs = dict(
            text_inputs=context_inputs_text, 
            face_inputs=self.get_face_inputs(relevant_passages), 
            image_inputs=self.get_image_inputs(relevant_passages)
        )

        # make n_irrelevant_passages by shifting the images of relevant passages
        n_irrelevant_passages = self.M-self.n_relevant_passages
        if n_irrelevant_passages > 0:
            # duplicate relevant text
            for k, v in context_inputs["text_inputs"].items():
                context_inputs["text_inputs"][k] = torch.tile(v, (self.M, 1))
            # shift relevant images
            for k, v in context_inputs['image_inputs'].items():
                shifted_input, shifted_mask = [v['input']], [v['attention_mask']]
                for shift in range(n_irrelevant_passages):
                    # shift along axis 0 (batch axis)
                    shifted_input.append(torch.roll(v['input'], shift+1, 0))
                    shifted_mask.append(torch.roll(v['attention_mask'], shift+1, 0))
                # cat along axis 0 (batch axis)
                v['input'] = torch.cat(shifted_input, 0)
                v['attention_mask'] = torch.cat(shifted_mask, 0)
            # shift relevant faces
            shifted_faces, shifted_boxes = [context_inputs['face_inputs']["face"]], [context_inputs['face_inputs']["bbox"]]
            shifted_mask = [context_inputs['face_inputs']['attention_mask']]
            for shift in range(n_irrelevant_passages):
                # shift along axis 0 (batch axis)
                shifted_faces.append(torch.roll(context_inputs['face_inputs']["face"], shift+1, 0))
                shifted_boxes.append(torch.roll(context_inputs['face_inputs']["bbox"], shift+1, 0))
                shifted_mask.append(torch.roll(context_inputs['face_inputs']["attention_mask"], shift+1, 0))
            # cat along axis 0 (batch axis)
            context_inputs['face_inputs']["face"] = torch.cat(shifted_faces, 0)
            context_inputs['face_inputs']["bbox"] = torch.cat(shifted_boxes, 0)
            context_inputs['face_inputs']['attention_mask'] = torch.cat(shifted_mask, 0)

        # wrap it up
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch


class MultiPassageBERTTrainer(QuestionAnsweringTrainer):
    """
    Specific for RC, more precisely MultiPassageBERT
    (will I manage to code an extra-level of abstraction, e.g. ReadingComprehensionTrainer?)

    Parameters
    ----------
    *args, **kwargs: additional arguments are passed to QuestionAnsweringTrainer
    max_n_answers: int, optional
        The answer might be found several time in the same passage, this is a threshold to enable batching
        Defaults to 10.
    ignore_keys: List[str], optional
        List of keys to remove from the batch before feeding it to the model
        (data not used by the model but necessary for evaluation)
        Defaults to ['answer_strings']
    train_original_answer_only: bool, optional
        Whether the model should be trained to predict only the original answer (default)
        or all alternative answers (with the only limit of max_n_answers)
        This has no effect on the evaluation (where all alternative answers are always considered)
    """
    def __init__(self, *args, max_n_answers=10, ignore_keys=['answer_strings'], train_original_answer_only=True, oracle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_n_answers = max_n_answers
        self.ignore_keys = ignore_keys
        self.train_original_answer_only = train_original_answer_only
        self.oracle = oracle
        if self.oracle:
            self.prediction_file_name = "oracle_predictions.json"
            self.metrics_file_name = "oracle_metrics.json"
            if self.n_relevant_passages != self.M:
                warnings.warn(f"Oracle mode. Setting n_relevant_passages={self.M}")
                self.n_relevant_passages = self.M

        # FIXME isn't there a more robust way of defining data_collator as the method collate_fn ?
        self.data_collator = self.collate_fn

    def get_eval_passages(self, item):
        """Keep the top-M passages retrieved by the IR"""
        indices = item[self.search_key+"_indices"][: self.M]
        scores = item[self.search_key+"_scores"][: self.M]
        return self.kb.select(indices), scores

    def get_answer_position(self, batch, answers, answer_mask):
        """Adapted from DPR"""
        start_positions, end_positions = torch.zeros_like(answer_mask), torch.zeros_like(answer_mask)
        for j, (input_ids, answer) in enumerate(zip(batch['input_ids'], answers)):
            L = input_ids.size(-1)
            answer_starts, answer_ends = [], []
            for a in answer:
                answer_len = a.size(0)
                enough = False
                for i in range(L-answer_len+1):
                    if (a == input_ids[i: i+answer_len]).all():
                        start, end = i, i+answer_len-1
                        if start not in answer_starts and end not in answer_ends:
                            answer_starts.append(start)
                            answer_ends.append(end)
                            if len(answer_starts) >= self.max_n_answers:
                                enough = True
                                break
                if enough:
                    break
            for i, (start, end) in enumerate(zip(answer_starts, answer_ends)):
                start_positions[j, i] = start
                end_positions[j, i] = end
                # un-mask answer
                answer_mask[j, i] = 1
        start_positions = start_positions.view(-1, self.M, self.max_n_answers)
        end_positions = end_positions.view(-1, self.M, self.max_n_answers)
        answer_mask = answer_mask.view(-1, self.M, self.max_n_answers)
        batch.update(dict(start_positions=start_positions, end_positions=end_positions, answer_mask=answer_mask))
        return batch

    def collate_fn(self, items):
        """
        Collate batch so that each question is associate with n_relevant_passages and M-n irrelevant ones.
        Also tokenizes input strings

        Returns (a dict of)
        -------------------
        input_ids: Tensor[int]
            shape (N * M, L)
        start_positions, end_positions: Tensor[int]
            shape (N, M, max_n_answers)
        answer_mask: Tensor[int]
            shape (N, M, max_n_answers)
        passage_scores: Tensor[float], optional
            shape (N * M)
            only in evaluation mode
        **kwargs: more tensors depending on the tokenizer, e.g. attention_mask
        """
        questions, passages = [], []
        answers, answer_strings = [], []
        passage_scores = []
        N = len(items)
        answer_mask = torch.zeros((N*self.M, self.max_n_answers), dtype=torch.long)
        for i, item in enumerate(items):
            # N. B. seed is set in Trainer
            questions.extend([item['input']]*self.M)

            # oracle -> use only relevant passages
            if (self.args.do_eval or self.args.do_predict) and not self.oracle:
                passage, score = self.get_eval_passages(item)
                passage = passage['passage']
                passage_scores.extend(score)
                if len(score) < self.M:
                    passage_scores.extend([0]*(self.M-len(score)))
            else:
                relevant_passage, irrelevant_passage = self.get_training_passages(item)
                passage = relevant_passage['passage'] + irrelevant_passage['passage']

            passages.extend(passage)
            # all passages have at least 1 non-masked answer (set to 0 for irrelevant passages)
            answer_mask[i*self.M: i*self.M+len(passage), 0] = 1
            # except for padding passages
            if len(passage) < self.M:
                passages.extend(['']*(self.M-len(passage)))

            original_answer = item['output']['original_answer']
            # avoid processing the same answer twice
            answer = item['output']['answer']
            answer_strings.extend([answer]*self.M)
            # beware this create a discrepancy between answer_strings and answers (tokens)
            # evaluation should always be done using answer_strings
            if self.train_original_answer_only:
                answer = [original_answer]
            else:
                if self.tokenizer.do_lower_case:
                    original_answer = original_answer.lower()
                    answer = list({a.lower() for a in answer} - {original_answer})
                # but ensure the original answer is still the first to be processed
                answer = [original_answer] + answer
            answer = self.tokenizer(answer,
                                    add_special_tokens=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False)['input_ids']
            answer = [torch.tensor(a, dtype=torch.long) for a in answer]
            answers.extend([answer]*self.M)
        batch = self.tokenizer(*(questions, passages), **self.tokenization_kwargs)
        batch = self.get_answer_position(batch, answers, answer_mask)
        batch['answer_strings'] = answer_strings
        if passage_scores:
            batch['passage_scores'] = torch.tensor(passage_scores)

        return batch

    def _prepare_inputs(self, inputs: dict) -> dict:
        """remove all keys not used by the model but necessary for evaluation before returning Trainer._prepare_inputs"""
        for k in self.ignore_keys:
            if k not in inputs:
                warnings.warn(f"Didn't find {k} in inputs")
                continue
            inputs.pop(k)
        return super()._prepare_inputs(inputs)

    def log_probs_to_answers(self, predictions, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        _, _, start_log_probs, end_log_probs = predictions
        passage_indices, start_indices, end_indices = get_best_spans(start_probs=np.exp(start_log_probs),
                                                                     end_probs=np.exp(end_log_probs),
                                                                     **kwargs)
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.tokenizer.batch_decode(answers, skip_special_tokens=True)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: list = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Same as Trainer.evaluation_loop but does not truncate output to the size of the dataset because
        there is M passages per question so the output is M times the size of the dataset

        Also gather input_ids instead of labels in order to recover the tokens from the model's span start and end probabilities
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        print(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/input_ids on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        input_ids_host = None
        passage_scores_host = None
        # losses/preds/input_ids on CPU (final containers)
        all_losses = None
        all_preds = None
        all_input_ids = None
        all_passage_scores = None
        all_answers = []

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            answer_strings = inputs.get('answer_strings')
            if answer_strings is not None:
                all_answers.extend(answer_strings)
            passage_score = inputs.get('passage_scores')
            if passage_score is not None:
                passage_scores_host = passage_score if passage_scores_host is None else torch.cat((passage_scores_host, passage_score), dim=0)

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            input_ids = self._pad_across_processes(inputs['input_ids'])
            input_ids = self._nested_gather(input_ids)
            input_ids_host = input_ids if input_ids_host is None else nested_concat(input_ids_host, input_ids, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                input_ids = nested_numpify(input_ids_host)
                all_input_ids = (
                    input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
                )
                if passage_scores_host is not None:
                    passage_scores = nested_numpify(passage_scores_host)
                    all_passage_scores = passage_scores if all_passage_scores is None else nested_concat(all_passage_scores, passage_scores, padding_index=0)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, input_ids_host, passage_scores_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if input_ids_host is not None:
            input_ids = nested_numpify(input_ids_host)
            all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
        if passage_scores_host is not None:
            passage_scores = nested_numpify(passage_scores_host)
            all_passage_scores = passage_scores if all_passage_scores is None else nested_concat(all_passage_scores, passage_scores, padding_index=0)

        # reshape like (N, M, L) to ease further processing
        if all_preds is not None:
            all_preds = tuple(pred.reshape(num_samples, self.M, -1) for pred in all_preds)
        if all_input_ids is not None:
            all_input_ids = all_input_ids.reshape(num_samples, self.M, -1)
        if all_passage_scores is not None:
            all_passage_scores = all_passage_scores.reshape(num_samples, self.M)
        if all_answers:
            all_answers = [all_answers[i] for i in range(0, len(all_answers), self.M)]
            assert len(all_answers) == num_samples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_input_ids is not None and all_answers:
            # 1. raw predictions from scores spans
            predictions = self.log_probs_to_answers(all_preds, all_input_ids)
            predictions, references = format_predictions_for_squad(predictions, all_answers)
            metrics = self.compute_metrics(predictions=predictions, references=references)
            # 2. weighted predictions
            if all_passage_scores is not None:
                weighted_predictions = self.log_probs_to_answers(all_preds, all_input_ids, weights=all_passage_scores)
                weighted_predictions, references = format_predictions_for_squad(weighted_predictions, all_answers)
                for k, v in self.compute_metrics(predictions=weighted_predictions, references=references).items():
                    metrics['weighted_'+k] = v
        else:
            metrics = {}
            predictions = all_preds

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=predictions, label_ids=None, metrics=metrics, num_samples=num_samples)


def get_checkpoint(resume_from_checkpoint: str, *args, **kwargs):
    if args or kwargs:
        warnings.warn(f"ignoring additional arguments:\n{args}\n{kwargs}")
    cpt = Path(resume_from_checkpoint)
    # weird trick to glob using pathlib
    resume_from_checkpoints = list(cpt.parent.glob(cpt.name))
    return resume_from_checkpoints


def subsample_dataset(dataset, num_shards):
    dataset = dataset.shuffle(seed=0)
    return dataset.shard(num_shards, 0)


def instantiate_trainer(trainee, trainer_class="MultiPassageBERTTrainer", debug=False, 
                        train_dataset=None, eval_dataset=None, metric='squad', 
                        training_kwargs={}, callbacks_args=[], 
                        train_shards=None, eval_shards=None, **kwargs):
    """Additional arguments are passed to Trainer"""
    # debug (see torch.autograd.detect_anomaly)
    set_detect_anomaly(debug)

    # data
    if train_dataset is not None:
        train_dataset = load_from_disk(train_dataset)
        if train_shards is not None:
            train_dataset = subsample_dataset(train_dataset, train_shards)
    if eval_dataset is not None:
        eval_dataset = load_from_disk(eval_dataset)
        if eval_shards is not None:
            eval_dataset = subsample_dataset(eval_dataset, eval_shards)

    # training
    # revert the post-init that overrides do_eval
    do_eval = training_kwargs.pop('do_eval', False)
    training_args = TrainingArguments(**training_kwargs)
    training_args.do_eval = do_eval

    # metrics come in priority from meerqat.train.metrics
    if metric is not None:
        compute_metrics = getattr(metric_functions, metric, None)
        # or from HF's datasets
        if compute_metrics is None:
            metric = load_metric(metric)
            compute_metrics = metric.compute
    else:
        compute_metrics = None

    TrainerClass = getattr(sys.modules[__name__], trainer_class)
    trainer = TrainerClass(model=trainee, args=training_args,
                           train_dataset=train_dataset, eval_dataset=eval_dataset,
                           compute_metrics=compute_metrics, **kwargs)
    # training callbacks
    for callback in callbacks_args:
        CallbackClass = getattr(trainer_callback, callback.pop("Class"))
        trainer.add_callback(CallbackClass(**callback))

    return trainer, training_args


if __name__ == "__main__":
    logger.debug(f"entering main {max_memory_usage(human=True)}")
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = load_pretrained_in_kwargs(json.load(file))

    logger.debug(f"after loading pre-trained models {max_memory_usage(human=True)}")

    verbosity = config.pop("verbosity", None)
    if verbosity is not None:
        t_logging.set_verbosity(verbosity)
        logger.setLevel(verbosity)

    checkpoint = config.pop("checkpoint", {})
    trainer, training_args = instantiate_trainer(**config)
    device = trainer.args.device
    logger.debug(f"after instantiating trainer {max_memory_usage(human=True)}")
    if training_args.do_train:
        trainer.train(**checkpoint)
    elif training_args.do_eval:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Evaluation"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path, map_location=device)
            trainer._load_state_dict_in_model(state_dict)

            # optionally load trainer state for better logging
            trainer_state = resume_from_checkpoint/"trainer_state.json"
            if trainer_state.is_file():
                trainer.state = TrainerState.load_from_json(trainer_state)
            else:
                warnings.warn("couldn't load trainer state, TB logging might use an inappropriate step")
            metrics = trainer.evaluate()
            trainer.write_metrics(metrics, resume_from_checkpoint)
    elif training_args.do_predict:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path, map_location=device)
            trainer._load_state_dict_in_model(state_dict)

            # run model on evaluation dataset
            prediction_output = trainer.predict(trainer.eval_dataset)
            trainer.write_metrics(prediction_output.metrics, resume_from_checkpoint)
            trainer.write_predictions(prediction_output.predictions, resume_from_checkpoint)
    else:
        warnings.warn("Did nothing except instantiate the trainer, "
                      "you probably want to set do_train, do_eval or do_predict to True"
                      f"see {training_args.__doc__}")
