# -*- coding: utf-8 -*-
"""Classes to format data in proper batches to train models"""
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict
#import ranx

import pytorch_lightning as pl

from ..data.loading import get_pretrained, verbose_load_from_disk
from ..models.utils import debug_shape


class DataModule(pl.LightningDataModule):
    """
    Base class for all data modules. 
    It has a tokenizer and handles dataset loading with train/validation/test subsets.
    
    Parameters
    ----------
    tokenizer_class: str
        Name of a transformers.PreTrainedTokenizer subclass
    tokenizer_name_or_path: str
        see transformers.PreTrainedTokenizer.from_pretrained
    dataset_path: str, optional
        Path to a DatasetDict that should have 'train', 'validation' and 'test' subsets.
        Alternatively, you can specify those using the dedicated variables.
    train_path, validation_path, test_path: str, optional
    train_batch_size, eval_batch_size: int, optional
    tokenization_kwargs: dict, optional
        To be passed to self.tokenizer
    """
    def __init__(self, tokenizer_class, tokenizer_name_or_path, 
                 dataset_path=None, train_path=None, validation_path=None, test_path=None, 
                 train_batch_size=8, eval_batch_size=8, tokenization_kwargs=None):
        super().__init__()
        self.tokenizer = get_pretrained(tokenizer_class, pretrained_model_name_or_path=tokenizer_name_or_path)
        self.dataset_path = dataset_path
        self.train_path = train_path        
        self.validation_path = validation_path
        self.test_path = test_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        # useful in some corner-cases like ICT. False for every other data modules
        self.shuffle_eval = False
        default_tokenization_kwargs = dict(return_tensors='pt', padding='longest', truncation=True)
        if tokenization_kwargs is not None:
            default_tokenization_kwargs.update(tokenization_kwargs)
        self.tokenization_kwargs = default_tokenization_kwargs
        
    def setup(self, stage=None):
        if self.dataset_path is None:
            self.dataset = DatasetDict()
            for subset in ['train', 'validation', 'test']:
                subset_path = getattr(self, subset+'_path', None)
                if subset_path is not None:
                    self.dataset[subset] = verbose_load_from_disk(subset_path)
        else:
            self.dataset = verbose_load_from_disk(self.dataset_path)
        
    def train_dataloader(self):    
        if 'train' not in self.dataset:
            return None    
        return DataLoader(
            self.dataset['train'], 
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        if 'validation' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['validation'], 
            batch_size=self.eval_batch_size, 
            collate_fn=self.collate_fn,
            shuffle=self.shuffle_eval
        )

    def test_dataloader(self):
        if 'test' not in self.dataset:
            return None
        return DataLoader(
            self.dataset['test'], 
            batch_size=self.eval_batch_size, 
            collate_fn=self.collate_fn,
            shuffle=self.shuffle_eval
        )


class PreComputedImageFeatures:
    """
    Helper to format image features in nice square Tensors expected by mm models.
    
    Parameters
    ----------
    config_class: str
        Name of a subclass of MMConfig
    config_path: str
    """
    def __init__(self, config_class, config_path):
        config = get_pretrained(config_class, pretrained_model_name_or_path=config_path)
        self.n_faces = config.n_faces        
        self.image_embeddings_keys = config.image_kwargs.keys()
        self.image_dims = {}
        for name in self.image_embeddings_keys:
            image_dim = config.image_kwargs[name]["input_dim"]
            self.image_dims[name] = image_dim
        self.face_dim = config.face_kwargs['face_dim']
        self.bbox_dim = config.face_kwargs['bbox_dim']

    def get_face_inputs(self, items):
        """
        Formats pre-computed face features in nice square tensors.
        
        Returns
        -------
        face_inputs: dict[str, Tensor]
            {
               * face: Tensor(batch_size, self.n_faces, self.face_dim)
               * bbox: Tensor(batch_size, self.n_faces, self.bbox_dim)
               * attention_mask: Tensor(batch_size, self.n_faces)
            }
        """
        # trim or pad, and convert to tensor
        face_embeddings = torch.zeros((len(items), self.n_faces, self.face_dim))
        face_boxes = torch.zeros((len(items), self.n_faces, self.bbox_dim))
        # 0=masked, 1=not masked
        attention_mask = torch.zeros((len(items), self.n_faces), dtype=torch.long)
        if self.n_faces == 0:
            return {
                "face": face_embeddings,
                "bbox": face_boxes,
                "attention_mask": attention_mask
            }
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
        """
        Formats pre-computed full-image features in nice square tensors.
        
        Returns
        -------
        image_inputs: dict[str, dict[str,Tensor]]
            one key per image feature
            {
               * input: Tensor(batch_size, ?)
               * attention_mask: Tensor(batch_size, )
            }
        """
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
    
    
class QuestionAnsweringDataModule(DataModule):
    """
    Base class for Question Answering. Should work for both IR and RC.
    
    The core idea is that it relies on a Knowledge Base (KB) 
    to retrieve relevant and irrelevant passages for the questions in the dataset.
    
    For multimodal models, it can also handle pre-computed image features stored in image_kb
    using PreComputedImageFeatures
    
    We need to create the batch of questions and passages on-the-fly
    The inputs should be shaped like (N * M, L), where:
        * N - number of distinct questions (equal to the batch size)
        * M - number of passages per question in a batch
        * L - sequence length

    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to DataModule
    kb: str
        path towards the knowledge base (Dataset) used to get the passages    
    image_kb: str, optional
        Path to the KB that holds pre-computed image features
        Can be mapped from kb using kb['index']
    image_features_kwargs: dict, optional
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
    """
    def __init__(self, *args, kb, image_kb=None, image_features_kwargs={}, 
                 M=24, n_relevant_passages=1, search_key='search', **kwargs):
        super().__init__(*args, **kwargs)
        self.kb = verbose_load_from_disk(kb)
        if image_kb is not None:
            self.image_kb = verbose_load_from_disk(image_kb)
            self.image_features = PreComputedImageFeatures(**image_features_kwargs)
            self.padding_passage = [{'passage': ''}]
        else:
            self.image_kb = None
            self.padding_passage = ['']
        self.M = M
        assert n_relevant_passages <= M
        self.n_relevant_passages = n_relevant_passages
        self.search_key = search_key    
                         
    def add_image_features(self, passages):
        """Add image features to passages from self.image_kb"""
        if len(passages) < 1:
            return passages
        features = ({"face_box", "face_embedding"} | self.image_features.image_embeddings_keys)
        batch = {'index': [], 'passage': []}
        for passage in passages:
            batch['index'].append(passage['index'])
            batch['passage'].append(passage['passage'])
        subset = self.image_kb.select(batch['index'])
        for feature in features:
            batch.setdefault(feature, subset[feature])
        # dict of list to list of dict
        output = []
        for values in zip(*batch.values()):
            output.append({k: v for k, v in zip(batch.keys(), values)})
        return output

    def get_training_passages(self, item):
        """
        Parameters
        ----------
        item: dict
            item (e.g. question) from self.train_dataset or self.eval_dataset.
        
        Returns
        -------
        relevant_passages, irrelevant_passages: list[dict]
            List of relevant and irrelevant passages selected from self.kb
            according to:
                - self.n_relevant_passages
                - self.M
                - self.search_key
        """
        # get passages from kb wrt search_key
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
        
        # multimodal vs. text-only mode
        if self.image_kb is None:
            relevant_passages = relevant_passages['passage']
            irrelevant_passages = irrelevant_passages['passage']
        else:
            relevant_passages = self.add_image_features(relevant_passages)
            irrelevant_passages = self.add_image_features(irrelevant_passages)
        return relevant_passages, irrelevant_passages  
            

class BiEncoderDataModule(QuestionAnsweringDataModule):        
    def collate_fn(self, items):
        """
        Collate batch so that each question is associate with n_relevant_passages and M-n irrelevant ones.
        Also tokenizes input strings

            * N - number of questions in a batch
            * M - number of passages per questions
            * d - dimension of the model/embeddings

        Returns (a dict of)
        -------------------
        question_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N, L)
            **kwargs: 
                more tensors depending on the tokenizer, e.g. attention_mask
        context_inputs: dict[torch.LongTensor]
            input_ids: torch.LongTensor
                shape (N*M, L)
                The first N rows correspond to the relevant contexts for the N questions
                The rest N*(M-1) rows are irrelevant contexts for all questions.
            **kwargs: 
                idem
        """
        n_irrelevant_passages = self.M-self.n_relevant_passages
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        for i, item in enumerate(items):
            relevant_passage, irrelevant_passage = self.get_training_passages(item)
            if len(relevant_passage) < 1:
                relevant_passage = self.padding_passage
                # FIXME hardcode -100 but this is dependent with BiEncoder that uses NLLLoss
                labels.append(-100)
            else:
                labels.append(i)
            if len(irrelevant_passage) < n_irrelevant_passages:
                irrelevant_passage.extend(self.padding_passage*(n_irrelevant_passages-len(irrelevant_passage)))
            questions.append(item['input'])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)

        # tokenize questions
        question_inputs_text = self.tokenizer(questions, **self.tokenization_kwargs)
        # concatenate passages and tokenize
        all_passages = relevant_passages + irrelevant_passages
        if self.image_kb is None:
            all_passages_text = all_passages
        else:
            all_passages_text = [p['passage'] for p in all_passages]
        context_inputs_text = self.tokenizer(all_passages_text, **self.tokenization_kwargs)
        
        # multimodal vs. text-only
        if self.image_kb is None:
            question_inputs = question_inputs_text
            context_inputs = context_inputs_text
        else:
            # get image features, for both questions and passages
            question_inputs = dict(
                text_inputs=question_inputs_text, 
                face_inputs=self.image_features.get_face_inputs(items), 
                image_inputs=self.image_features.get_image_inputs(items)
            )
            context_inputs = dict(
                text_inputs=context_inputs_text, 
                face_inputs=self.image_features.get_face_inputs(all_passages), 
                image_inputs=self.image_features.get_image_inputs(all_passages)
            )
        
        # wrap it up
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch
        

def filter_rels(dataset, search_key):
    # TODO
    """
    Filter out questions of the dataset without any relevant passages.
    
    
    Parameters
    ----------
    dataset: Dataset
    search_key: str
        see QuestionAnsweringTrainer
    
    Returns
    -------
    dataset: Dataset
        With at least one relevant passage for all questions.
    """
    before_len = len(dataset)
    dataset = dataset.filter(lambda item: len(item[f"{search_key}_provenance_indices"]) > 0)
    after_len = len(dataset)
    print(f"Filtered the dataset with empty '{search_key}_provenance_indices' from {before_len} to {after_len} items")
    return dataset