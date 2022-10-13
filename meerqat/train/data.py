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
        
    
class ICT(DataModule):
    """
    Extends the Inverse Cloze Task (ICT, [2]_) to multimodal documents.
    Given a wikipedia section, one sentence is considered as a pseudo-question and the nearby sentences as a relevant passage.
    In this multimodal setting, we also consider the image of the section in the query and the infobox/main image of the article in the visual passage.
    
    The only point in common with QuestionAnsweringDataModule is the use of PreComputedImageFeatures
    
    Parameters
    ----------
    *args, **kwargs: 
        additional arguments are passed to DataModule
    image_features_kwargs: dict
    sentences_per_target: int, optional
        Number of sentences in the target passages
    n_hard_negatives: int, optional
        Synthesize hopefully-hard negatives by permuting images in the batch n times
        Defaults to only random in-batch negatives
    prepend_title: bool, optional
        Whether to preprend the title of the article to the target passage
    text_mask_rate: float, optional
        Rate at which the pseudo-question is masked in the target passage
    image_mask_rate: float, optional
        Rate at which the infobox image is used as target (keep input image otherwise)

    References
    ----------
    .. [2] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent Retrieval for Weakly Supervised Open Domain Question Answering. 
       In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 
       pages 6086–6096, Florence, Italy. Association for Computational Linguistics.
    """
    def __init__(self, *args, image_features_kwargs, sentences_per_target=4, n_hard_negatives=0,
                 prepend_title=False, text_mask_rate=1.0, image_mask_rate=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_features = PreComputedImageFeatures(**image_features_kwargs)
        self.sentences_per_target = sentences_per_target
        self.n_hard_negatives = n_hard_negatives
        self.prepend_title = prepend_title
        self.text_mask_rate = text_mask_rate
        self.image_mask_rate = image_mask_rate
        # the WIT dataset groups wikipedia sections by article 
        # so in-batch negatives may get very difficult or even false positives if we don’t shuffle
        self.shuffle_eval = True

    def get_pseudo_question(self, item):
        """
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
        # standard ICT: remove sentence from its context
        if np.random.rand() < self.text_mask_rate:
            target = [s['text'] for s in sentences[i-shift: i]+sentences[i+1: i+1+n-shift]]
        # robustness trick: keep the sentence in the context so that the model learns lexical overlap
        else:
            target = [s['text'] for s in sentences[i-shift: i+1+n-shift]]
            
        if self.prepend_title:
            target.insert(0, self.tokenizer.sep_token)
            target.insert(0, item['title'])
        target = dict(text=" ".join(target))  
        
        # standard MICT: use the contextual image as target
        if np.random.rand() < self.image_mask_rate:
            context_image_key = "context_"
        # robustness trick: use the same image in query/target so that the model keeps image information
        else:
            context_image_key = ""
        # rename context image features
        for k in ({"face_box", "face_embedding"} | self.image_features.image_embeddings_keys):
            target[k] = item.get(f"{context_image_key}{k}")
        return query, target

    def collate_fn(self, items):
        questions, relevant_passages, labels = [], [], []
        for i, item in enumerate(items):
            query, relevant_passage = self.get_pseudo_question(item)
            labels.append(i)
            questions.append(query)
            relevant_passages.append(relevant_passage)

        question_inputs_text = self.tokenizer([q['text'] for q in questions], **self.tokenization_kwargs)
        context_inputs_text = self.tokenizer([p['text'] for p in relevant_passages], **self.tokenization_kwargs)
        # get image features, for both questions and passages
        question_inputs = dict(
            text_inputs=question_inputs_text, 
            face_inputs=self.image_features.get_face_inputs(items), 
            image_inputs=self.image_features.get_image_inputs(items)
        )
        context_inputs = dict(
            text_inputs=context_inputs_text, 
            face_inputs=self.image_features.get_face_inputs(relevant_passages), 
            image_inputs=self.image_features.get_image_inputs(relevant_passages)
        )

        # make self.n_hard_negatives by shifting the images of relevant passages
        if self.n_hard_negatives > 0:
            # duplicate relevant text
            for k, v in context_inputs["text_inputs"].items():
                context_inputs["text_inputs"][k] = torch.tile(v, (self.n_hard_negatives+1, 1))
            # shift relevant images
            for k, v in context_inputs['image_inputs'].items():
                shifted_input, shifted_mask = [v['input']], [v['attention_mask']]
                for shift in range(self.n_hard_negatives):
                    # shift along axis 0 (batch axis)
                    shifted_input.append(torch.roll(v['input'], shift+1, 0))
                    shifted_mask.append(torch.roll(v['attention_mask'], shift+1, 0))
                # cat along axis 0 (batch axis)
                v['input'] = torch.cat(shifted_input, 0)
                v['attention_mask'] = torch.cat(shifted_mask, 0)
            # shift relevant faces
            shifted_faces, shifted_boxes = [context_inputs['face_inputs']["face"]], [context_inputs['face_inputs']["bbox"]]
            shifted_mask = [context_inputs['face_inputs']['attention_mask']]
            for shift in range(self.n_hard_negatives):
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