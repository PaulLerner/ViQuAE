# -*- coding: utf-8 -*-
"""Classes to format data in proper batches to train models"""
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk, DatasetDict
#import ranx

import pytorch_lightning as pl

from ..data.loading import get_pretrained
from ..models.utils import debug_shape


class DataModule(pl.LightningDataModule):
    """
    Base class for all data modules. Provides basic stuff to behave similarly to transformers.
    
    Parameters
    ----------
    dataset_path: str, optional
        Path to a DatasetDict that should have 'train', 'validation' and 'test' subsets.
        Alternatively, you can specify those using the dedicated variables.
    train_path, validation_path, test_path: str, optional
    train_batch_size, eval_batch_size: int, optional
    """
    def __init__(self, dataset_path=None, train_path=None, validation_path=None, test_path=None, train_batch_size=8, eval_batch_size=8):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_path = train_path        
        self.validation_path = validation_path
        self.test_path = test_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.shuffle_eval = False
        
    def setup(self, stage=None):
        if self.dataset_path is None:
            self.dataset = DatasetDict()
            for subset in ['train', 'validation', 'test']:
                subset_path = getattr(self, subset+'_path', None)
                if subset_path is not None:
                    print(f"loading {subset_path}")
                    self.dataset[subset] = load_from_disk(subset_path)
        else:
            self.dataset = load_from_disk(self.dataset_path)
        print(self.dataset)
        
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


# FIXME: are those really data modules ? maybe switch to data collator to avoid multiple inheritance
class QuestionAnsweringDataModule(DataModule):
    """
    Base class for Question Answering. Should work for both IR and RC.
    Overrides some methods because we need to create the batch of questions and passages on-the-fly
    Because the inputs should be shaped like (N * M, L), where:
        * N - number of distinct questions
        * M - number of passages per question in a batch
        * L - sequence length

    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to DataModule
    tokenizer: transformers.PreTrainedTokenizer
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
    def __init__(self, *args, tokenizer_class, tokenizer_name_or_path, kb=None, M=24, n_relevant_passages=1, search_key='search', tokenization_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = get_pretrained(tokenizer_class, pretrained_model_name_or_path=tokenizer_name_or_path)
        if kb is not None:
            print(f"loading {kb}")
            self.kb = load_from_disk(kb)
            print(self.kb)
        else:
            self.kb = None
        self.M = M
        assert n_relevant_passages <= M
        self.n_relevant_passages = n_relevant_passages
        self.search_key = search_key
        default_tokenization_kwargs = dict(return_tensors='pt', padding='longest', truncation=True)
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        default_tokenization_kwargs.update(tokenization_kwargs)
        self.tokenization_kwargs = default_tokenization_kwargs

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
            relevant_passage, irrelevant_passage = [p['passage'] for p in relevant_passage], [p['passage'] for p in  irrelevant_passage]
            if len(relevant_passage) < 1:
                relevant_passage = ['']
                # FIXME hardcode -100 but this is dependent with BiEncoder that uses NLLLoss
                labels.append(-100)
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