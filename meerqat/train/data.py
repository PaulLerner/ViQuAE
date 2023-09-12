# -*- coding: utf-8 -*-
"""
Classes to format data in proper batches to train models.
Also holds example generation methods such as Multimodal Inverse Cloze Task (ICT),
and dynamic examples based on passages retrieved from KB.
"""
import warnings
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict
import ranx

import pytorch_lightning as pl

from ..data.loading import get_pretrained, verbose_load_from_disk, load_image
from ..ir.metrics import find_valid_numerical_answers
from ..data.infoseek import QuestionType
from ..models.utils import debug_shape


def pad_and_cat(arrays, padding_index=-100):
    N, M, L = arrays[0].shape
    for array in arrays[1:]:
        n, m, l = array.shape
        assert m == M
        L = max(l, L)
        N += n
    result = np.full_like(arrays[0], padding_index, shape=(N, M, L))
    N = 0
    for array in arrays:
        n, _, l = array.shape
        result[N:N+n, :, :l] = array
        N += n
    return result
        

def keep_columns(dataset, columns):
    to_remove = [c for c in dataset.column_names if c not in columns]
    if to_remove:
        dataset = dataset.remove_columns(to_remove)
        print(f"Removed {to_remove} from the dataset:\n{dataset}")
    else:        
        print(f"Nothing to remove from the dataset:\n{dataset}")
    return dataset


# FIXME can we get rid of all these get_pretrained and automate the process in trainer.CLI?
class DataModule(pl.LightningDataModule):
    """
    Base class for all data modules. 
    It has a tokenizer and handles dataset loading with train/validation/test subsets.
    For multimodal models, it can also handle image features or pixels using ImageFormatter

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
    batch_size, train_batch_size, eval_batch_size: int, optional
        batch_size is needed to be able to tune it automatically using auto_scale_batch_size in Trainer
        It is overriden by train_batch_size, eval_batch_size 
        (if you want to use different batch sizes for training and evaluation)    
    M: int, optional
        Number of passages (relevant or irrelevant) per question in a batch
        Defaults to 24
    n_relevant_passages: int, optional
        Defaults to 1
    keep_dataset_columns: list, optional
        Keep only these features in the dataset (defaults to keep everything)
    tokenization_kwargs: dict, optional
        To be passed to self.tokenizer
    image_kwargs: dict, optional
        Passed to ImageFormatter. Optional for text-only models.
    loader_kwargs: dict, optional
        Passed to the data loaders (e.g. self.train_dataloader())
    dataset_format: dict, optional
        see Dataset.set_format
        Overrides keep_dataset_columns.
    input_key: str, optional
        Holds input text (e.g. question, caption), defaults to 'input'
    """
    def __init__(self, tokenizer_class, tokenizer_name_or_path, 
                 dataset_path=None, train_path=None, validation_path=None, test_path=None, 
                 batch_size=8, train_batch_size=None, eval_batch_size=None, 
                 M=24, n_relevant_passages=1, keep_dataset_columns=None,
                 tokenization_kwargs=None, image_kwargs={}, loader_kwargs={}, 
                 dataset_format=None, input_key='input'):
        super().__init__()
        self.tokenizer = get_pretrained(tokenizer_class, pretrained_model_name_or_path=tokenizer_name_or_path)
        self.dataset_path = dataset_path
        self.train_path = train_path        
        self.validation_path = validation_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.M = M
        self.n_relevant_passages = n_relevant_passages
        self.keep_dataset_columns = set(keep_dataset_columns) if keep_dataset_columns is not None else None
        self.dataset_format = dataset_format
        self.input_key = input_key
        
        # useful in some corner-cases like ICT. False for every other data modules
        self.shuffle_eval = False
        default_tokenization_kwargs = dict(
            return_tensors='pt', 
            padding='longest', 
            truncation=True, 
            return_overflowing_tokens=False
        )
        if tokenization_kwargs is not None:
            default_tokenization_kwargs.update(tokenization_kwargs)
        self.tokenization_kwargs = default_tokenization_kwargs
        self.image_formatter = ImageFormatter(**image_kwargs)
        self.loader_kwargs = loader_kwargs
        
    def setup(self, stage=None):
        if self.dataset_path is None:
            self.dataset = DatasetDict()
            for subset in ['train', 'validation', 'test']:
                subset_path = getattr(self, subset+'_path', None)
                if subset_path is not None:
                    self.dataset[subset] = verbose_load_from_disk(subset_path)
        else:
            self.dataset = verbose_load_from_disk(self.dataset_path)
        if self.dataset_format is not None:
            self.dataset.set_format(**self.dataset_format)
        elif self.keep_dataset_columns is not None:
            for name, subset in self.dataset.items():
                self.dataset[name] = keep_columns(subset, self.keep_dataset_columns)
            
    def train_dataloader(self):    
        if 'train' not in self.dataset:
            return None    
        # set here and not in __init__ so that it will be reset properly in Trainer.reset_train_dataloader,
        # which is called during auto_scale_batch_size
        batch_size = self.train_batch_size if self.train_batch_size is not None else self.batch_size
        return DataLoader(
            self.dataset['train'], 
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            **self.loader_kwargs
        )

    def val_dataloader(self):
        if 'validation' not in self.dataset:
            return None
        batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
        return DataLoader(
            self.dataset['validation'], 
            batch_size=batch_size, 
            collate_fn=self.collate_fn,
            shuffle=self.shuffle_eval,
            **self.loader_kwargs
        )

    def test_dataloader(self):
        if 'test' not in self.dataset:
            return None
        batch_size = self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
        return DataLoader(
            self.dataset['test'], 
            batch_size=batch_size, 
            collate_fn=self.collate_fn,
            shuffle=self.shuffle_eval,
            **self.loader_kwargs
        )
    

class ImageFormatter:
    """
    Helper to format image features (precomputed or pixels) in nice square Tensors expected by mm models.
    """
    def __init__(self, *args, precomputed=None, **kwargs):
        self.precomputed = precomputed
        if precomputed is None:
            # text-only
            assert not (args or kwargs), f"Maybe you forgot to set `precomputed=True|False`. Got:\n{args}\n{kwargs}"
        elif precomputed:
            self.image_features = PreComputedImageFeatures(*args, **kwargs)
        else:
            self.feature_extractor = get_pretrained(*args, **kwargs)
                 
    def format_pixels(self, items, image_key='image'):
        """Load images and convert to tensors while handling padded passages"""
        images, indices = [], []
        for i, item in enumerate(items):
            # in case of padding passage
            if image_key not in item:
                continue
            image = item[image_key]            
            # CrossModalDataModule: optionally sample a positive image in case of multiple reference images
            # e.g. several reference images in the KB, as in WIT or Mensink's EVQA
            if isinstance(image, list):
                image = np.random.choice(image)
            image = load_image(image)
            # trouble during loading. user is already warned
            if image is None:
                continue
            indices.append(i)
            images.append(image)
        
        # corner-case: only padding images
        if not images:
            size = self.feature_extractor.size
            pixel_values = torch.zeros(len(items), 3, size, size)
            return dict(pixel_values=pixel_values) 
        
        # resize and pad actual images using feature_extractor
        # N. B. this is three time slower than load_image (in cumulated time)
        images = self.feature_extractor(images, return_tensors="pt")
        b, c, h, w = images['pixel_values'].shape
        
        # opposite corner-case: no padding image, no need for all this trouble
        if b == len(items):
            return images
        
        # there are some padded images to handle
        pixel_values = torch.zeros(len(items), c, h, w)

        indices = torch.tensor(indices)
        pixel_values[indices] = images['pixel_values']
        output = dict(pixel_values=pixel_values)
        
        # pixel_mask exists for ViLT but not CLIP
        if 'pixel_mask' in images:
            pixel_mask = torch.zeros(len(items), h, w, dtype=torch.long)
            # at least one pixel should not be padded to avoid the following error:
    #       File "transformers/models/vilt/modeling_vilt.py", line 129, in <listcomp>                                               
    #           nn.functional.interpolate(                                                                                                                                                                             
    #       File "torch/nn/functional.py", line 3938, in interpolate                                                                
    #           return torch._C._nn.upsample_bilinear2d(input, output_size, align_corners, scale_factors)                                                                                                              
    #       RuntimeError: Input and output sizes should be greater than 0, but got input (H: 12, W: 12) output (H: 0, W: 0) 
            pixel_mask[:, 0, 0] = 1  
            pixel_mask[indices] = images['pixel_mask']   
            output['pixel_mask'] = pixel_mask
        
        return output
        
    def format_batch(self, text_inputs, items_or_passages, passages=None):
        """
        Parameters
        ----------
        text_inputs: dict[Tensor]
        items_or_passages: List[dict]
        passages: List[dict], optional
        """
        if self.precomputed is None:
            # text-only
            inputs = text_inputs
        elif self.precomputed:
            items_face_inputs = self.image_features.get_face_inputs(items_or_passages)            
            items_image_inputs = self.image_features.get_image_inputs(items_or_passages)
            
            # add a new dimension after batch, e.g. (batch_size, n_images, n_faces, face_dim) for faces
            if passages is not None:
                passage_face_inputs = self.image_features.get_face_inputs(passages)
                passage_image_inputs = self.image_features.get_image_inputs(passages)
                for k, v in passage_face_inputs.items():
                    items_face_inputs[k] = torch.cat((items_face_inputs[k], v), dim=1)
                for name, image in passage_image_inputs.items():
                    for k, v in image.items():
                        items_image_inputs[name][k] = torch.cat((items_image_inputs[name][k], v), dim=1)
                        
            inputs = dict(
                text_inputs=text_inputs,
                face_inputs=items_face_inputs, 
                image_inputs=items_image_inputs
            )
        else:
            items_pixels = self.format_pixels(items_or_passages)
            if passages is not None:
                passages_pixels = self.format_pixels(passages)
                items_pixels.update({f"passage_{k}": v for k, v in passages_pixels.items()})
            inputs = dict(
                **text_inputs,
                **items_pixels
            )
        return inputs
    
    
class PreComputedImageFeatures:
    """
    Helper to format image features in nice square Tensors expected by mm models.
    
    Parameters
    ----------
    config_class: str
        Name of a subclass of MMConfig
    config_path: str
    """
    def __init__(self, config_class, config_path, **kwargs):
        config = get_pretrained(config_class, pretrained_model_name_or_path=config_path, **kwargs)
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
        
        The extra dimension 1 stands for the number of images 
        (images are processed one by one and are concatenated in ImageFormatter)
        
        Returns
        -------
        face_inputs: dict[str, Tensor]
            {
               * face: Tensor(batch_size, 1, self.n_faces, self.face_dim)
               * bbox: Tensor(batch_size, 1, self.n_faces, self.bbox_dim)
               * attention_mask: Tensor(batch_size, 1, self.n_faces)
            }
        """
        # trim or pad, and convert to tensor
        face_embeddings = torch.zeros((len(items), 1, self.n_faces, self.face_dim))
        face_boxes = torch.zeros((len(items), 1, self.n_faces, self.bbox_dim))
        # 0=masked, 1=not masked
        attention_mask = torch.zeros((len(items), 1, self.n_faces), dtype=torch.long)
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
            face_embeddings[i, 0, : n_faces] = torch.tensor(face_embedding[: n_faces])
            bbox = item["face_box"]
            face_boxes[i, 0, : n_faces] = torch.tensor(bbox[: n_faces])
            attention_mask[i, 0, : n_faces] = 1

        face_inputs = {
            "face": face_embeddings,
            "bbox": face_boxes,
            "attention_mask": attention_mask
        }
        return face_inputs

    def get_image_inputs(self, items):
        """
        Formats pre-computed full-image features in nice square tensors.
        
        The extra dimension 1 stands for the number of images 
        (images are processed one by one and are concatenated in ImageFormatter)

        Returns
        -------
        image_inputs: dict[str, dict[str,Tensor]]
            one key per image feature
            {
               * input: Tensor(batch_size, 1, ?)
               * attention_mask: Tensor(batch_size, 1)
            }
        """
        image_inputs = {}
        for name in self.image_embeddings_keys: 
            features = torch.zeros(len(items), 1, self.image_dims[name])
            # 0=masked, 1=not masked
            attention_mask = torch.zeros(len(items), 1, dtype=torch.long)

            for i, item in enumerate(items):
                feature = item.get(name)
                # in case of padding passage
                if feature is None:
                    # keep zero-padding/mask
                    continue
                features[i, 0] = torch.tensor(feature)
                attention_mask[i, 0] = 1

            image_inputs[name] = dict(input=features, attention_mask=attention_mask)
        return image_inputs               
    
    
class CrossModalDataModule(DataModule):
    """
    Used for cross-modal retrieval (text-to-image or image-to-text) and optionally
    for joint cross-modal and image-image retrieval.
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to DataModule
    paired_image: str, optional
        If not None (default), should hold the key for the path to an image paired with 'image',
        so that a joint image-image contrastive loss may be applied in CrossModal(Trainee).
    deduplicate: bool, optional
        Will remove text (and paired_image) duplicates. 
        Defaults to False (assumes there are no duplicates).
    """
    def __init__(self, *args, paired_image=None,  deduplicate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.paired_image = paired_image
        self.shuffle_eval = True        
        self.deduplicate = deduplicate
        
        assert self.n_relevant_passages == 1
        assert self.M == 1

    def collate_fn(self, items):
        strings = [item[self.input_key] for item in items]
        text_inputs = self.tokenizer(strings, **self.tokenization_kwargs)
        batch = self.image_formatter.format_batch(text_inputs, items) 
        if self.deduplicate:
            _, where, labels = np.unique(strings, return_index=True, return_inverse=True)
            where, labels = torch.tensor(where), torch.tensor(labels)
            for k in batch.keys()-{'pixel_values'}:
                batch[k] = batch[k][where]
        else:
            labels = torch.arange(strings)
        if self.paired_image is not None:
            for k, v in self.image_formatter.format_pixels(items, image_key=self.paired_image).items():
                if self.deduplicate:
                    batch[f"paired_{k}"] = v[where]
                else:
                    batch[f"paired_{k}"] = v 
        batch['labels'] = labels
        return batch
        
    
class QuestionAnsweringDataModule(DataModule):
    """
    Base class for Question Answering. Should work for both IR and RC.
    
    The core idea is that it relies on a Knowledge Base (KB) 
    to retrieve relevant and irrelevant passages for the questions in the dataset.
        
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
    search_key: str, optional
        This column in the dataset suffixed by '_indices' and '_scores' should hold the result of information retrieval
        used during evaluation (e.g. the output of ir.search)
        Suffixed by "_provenance_indices" and "_irrelevant_indices" it should hold:
            1. the union of relevant search and provenance_indices
            2. irrelevant results from the search
        used during training (according to M and n_relevant_passages)
        Defaults to 'search'
    filter_train_rels: bool, optional     
    keep_kb_columns: list, optional
        Keep only these features in kb and image_kb (defaults to keep everything)
    kb_format, image_kb_format: dict, optional
        see Dataset.set_format
        Overrides keep_kb_columns.
    kb_input_key: str, optional
        Defaults to 'passage'
    """
    def __init__(self, *args, kb, image_kb=None, search_key='search', 
                 filter_train_rels=False, keep_kb_columns=None, 
                 kb_format=None, image_kb_format=None, kb_input_key='passage', **kwargs):
        super().__init__(*args, **kwargs)
        #TODO wiki.set_format('torch', ['clip-RN50'])
        self.kb = verbose_load_from_disk(kb)
        if kb_format is not None:
            self.kb.set_format(**kb_format)
        elif keep_kb_columns is not None:
            keep_kb_columns = set(keep_kb_columns)
            self.kb = keep_columns(self.kb, keep_kb_columns)
        if image_kb is not None:
            self.image_kb = verbose_load_from_disk(image_kb)            
            self.padding_passage = [{self.kb_input_key: ''}]
            if image_kb_format is not None:
                self.image_kb.set_format(**image_kb_format)
            elif keep_kb_columns is not None:
                self.image_kb = keep_columns(self.image_kb, keep_kb_columns)
        else:
            self.image_kb = None
            self.padding_passage = ['']
        self.search_key = search_key    
        if self.image_formatter.precomputed:
            self.add_image = self.add_image_features
        else:
            self.add_image = self.add_image_path
        self.filter_train_rels = filter_train_rels
        self.kb_input_key = kb_input_key
        
    def setup(self, stage=None):
        super().setup(stage=stage)
        if self.filter_train_rels and 'train' in self.dataset:
            self.filter_rels('train')

    def filter_rels(self, subset='train'):
        """
        Filter out questions of the dataset without any relevant passages.
        """
        before_len = len(self.dataset[subset])
        self.dataset[subset] = self.dataset[subset].filter(
            lambda item: len(item[f"{self.search_key}_provenance_indices"]) > 0, 
            new_fingerprint=f"{subset}_{self.search_key}_provenance_indices"
        )
        after_len = len(self.dataset[subset])
        print(f"Filtered {subset} dataset with empty '{self.search_key}_provenance_indices' from {before_len} to {after_len} items")
        
    def get_training_passages(self, item, with_scores=False):
        """
        Parameters
        ----------
        item: dict
            item (e.g. question) from self.train_dataset or self.eval_dataset.
        with_scores: bool, optional
            Also return the scores corresponding to the passages
            Defaults to False.
        
        Returns
        -------
        relevant_passages, irrelevant_passages: list[dict]
            List of relevant and irrelevant passages selected from self.kb
            according to:
                - self.n_relevant_passages
                - self.M
                - self.search_key
        relevant_scores: np.ndarray, optional
            Shape (self.n_relevant_passages, )
            Returned only if with_scores
        irrelevant_scores: np.ndarray, optional 
            Shape (self.M-self.n_relevant_passages, )
            Returned only if with_scores
        """
        assert self.n_relevant_passages <= self.M
        # get passages from kb wrt search_key
        relevant_passages, relevant_scores = [], []
        all_relevant_indices = item[self.search_key+"_provenance_indices"]
        n_relevant = min(len(all_relevant_indices), self.n_relevant_passages)
        if n_relevant > 0:
            i = np.arange(n_relevant)
            np.random.shuffle(i)
            relevant_indices = np.array(all_relevant_indices)[i]
            if with_scores:
                relevant_scores = np.array(item[self.search_key+"_provenance_scores"], dtype=np.float32)[i]
            relevant_passages = self.kb.select(relevant_indices)
        irrelevant_passages, irrelevant_scores = [], []
        all_irrelevant_indices = item[self.search_key+"_irrelevant_indices"]
        n_irrelevant = min(len(all_irrelevant_indices), self.M-self.n_relevant_passages)
        if n_irrelevant > 0:
            i = np.arange(n_irrelevant)
            np.random.shuffle(i)
            irrelevant_indices = np.array(all_irrelevant_indices)[i]
            if with_scores:
                irrelevant_scores = np.array(item[self.search_key+"_irrelevant_scores"], dtype=np.float32)[i]
            irrelevant_passages = self.kb.select(irrelevant_indices)
        elif n_relevant <= 0:
            warnings.warn(f"Didn't find any passage for question {item['id']}")
        
        # multimodal vs. text-only mode
        if self.image_kb is None:
            if relevant_passages:
                relevant_passages = relevant_passages[self.kb_input_key]
            if irrelevant_passages:
                irrelevant_passages = irrelevant_passages[self.kb_input_key]
        else:
            relevant_passages = self.add_image(list(relevant_passages))
            irrelevant_passages = self.add_image(list(irrelevant_passages))     
        if with_scores:
            return relevant_passages, irrelevant_passages, relevant_scores, irrelevant_scores
        else:
            return relevant_passages, irrelevant_passages
                    
    def add_image_features(self, passages):
        """
        Add image features to passages from image_kb
        
        Parameters
        ----------
        passages: List[dict]
        """
        if len(passages) < 1:
            return passages
        features = ({"face_box", "face_embedding"} | self.image_formatter.image_features.image_embeddings_keys)
        batch = {'index': [], self.kb_input_key: []}
        for passage in passages:
            batch['index'].append(passage['index'])
            batch[self.kb_input_key].append(passage[self.kb_input_key])
        subset = self.image_kb.select(batch['index'])
        for feature in features:
            batch.setdefault(feature, subset[feature])
        # dict of list to list of dict
        output = []
        for values in zip(*batch.values()):
            output.append({k: v for k, v in zip(batch.keys(), values)})
        return output
    
    def add_image_path(self, passages):
        """
        Add image path to passages from image_kb
        
        Parameters
        ----------
        passages: List[dict]
        """
        if len(passages) < 1:
            return passages
        for passage in passages:
            i = passage['index']
            passage.setdefault('image', self.image_kb[i]['image'])
        return passages   
        

class BiEncoderDataModule(QuestionAnsweringDataModule): 
    """
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to QuestionAnsweringDataModule
    passage_type_ids: bool, optional
        Pass token_type_ids=1 for passages (see BertTokenizer for details).
        This might be useful if you use a shared encoder to encode questions and passages.
        Defaults to False (by default you use different models to encode questions and passages).
    """
    def __init__(self, *args, passage_type_ids=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.passage_type_ids = passage_type_ids

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
        labels: torch.LongTensor
            shape (N, )
            Index of the relevant passage in context_inputs.
            Should be arange(N) except for padding passages.
        """        
        assert self.n_relevant_passages == 1
        n_irrelevant_passages = self.M-self.n_relevant_passages
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        for i, item in enumerate(items):
            relevant_passage, irrelevant_passage = self.get_training_passages(item)
            if len(relevant_passage) < 1:
                relevant_passage = self.padding_passage
                labels.append(self.trainer.lightning_module.loss_fct.ignore_index)
            else:
                labels.append(i)
            if len(irrelevant_passage) < n_irrelevant_passages:
                irrelevant_passage.extend(self.padding_passage*(n_irrelevant_passages-len(irrelevant_passage)))
            questions.append(item[self.input_key])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)

        # tokenize questions
        question_inputs_text = self.tokenizer(questions, **self.tokenization_kwargs)
        # concatenate passages and tokenize
        all_passages = relevant_passages + irrelevant_passages
        if self.image_kb is None:
            all_passages_text = all_passages
        else:
            all_passages_text = [p[self.kb_input_key] for p in all_passages]
        context_inputs_text = self.tokenizer(all_passages_text, **self.tokenization_kwargs)
        if self.passage_type_ids:
            context_inputs_text['token_type_ids'][context_inputs_text['attention_mask'].bool()] = 1
        
        # wrap it up
        question_inputs = self.image_formatter.format_batch(question_inputs_text, items)
        context_inputs = self.image_formatter.format_batch(context_inputs_text, all_passages)
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch
        
    
class JointBiEncoderAndClipDataModule(BiEncoderDataModule):
    def __init__(self, *args, cm_tokenizer_class, cm_tokenizer_name_or_path, cm_tokenization_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)       
        self.cm_tokenizer = get_pretrained(cm_tokenizer_class, pretrained_model_name_or_path=cm_tokenizer_name_or_path)
        default_tokenization_kwargs = self.tokenization_kwargs.copy()
        if cm_tokenization_kwargs is not None:
            default_tokenization_kwargs.update(cm_tokenization_kwargs)
        self.cm_tokenization_kwargs = default_tokenization_kwargs
        
    def collate_fn(self, items):           
        # TODO do not load/process for modules with weight=0
        assert self.n_relevant_passages == 1
        n_irrelevant_passages = self.M-self.n_relevant_passages
        questions, relevant_passages, irrelevant_passages, labels = [], [], [], []
        relevant_titles, irrelevant_titles = [], []
        for i, item in enumerate(items):
            relevant_passage, irrelevant_passage = self.get_training_passages(item)
            for p in relevant_passage:
                relevant_titles.append(p[self.kb_input_key][:p[self.kb_input_key].find('[SEP]')-1])
            if len(relevant_passage) < 1:
                relevant_passage = self.padding_passage
                labels.append(self.trainer.lightning_module.loss_fct.ignore_index)
                relevant_titles.append('')
            else:
                labels.append(i)
            for p in irrelevant_passage:
                irrelevant_titles.append(p[self.kb_input_key][:p[self.kb_input_key].find('[SEP]')-1])
            if len(irrelevant_passage) < n_irrelevant_passages:
                irrelevant_passage.extend(self.padding_passage*(n_irrelevant_passages-len(irrelevant_passage)))
                irrelevant_titles.extend(['']*(n_irrelevant_passages-len(irrelevant_passage)))
            questions.append(item[self.input_key])
            relevant_passages.extend(relevant_passage)
            irrelevant_passages.extend(irrelevant_passage)
        # tokenize questions
        question_inputs_text = self.tokenizer(questions, **self.tokenization_kwargs)
        # concatenate titles and tokenize 
        all_titles = relevant_titles + irrelevant_titles
        all_titles = self.cm_tokenizer(all_titles, **self.cm_tokenization_kwargs)        
        # concatenate passages and tokenize
        all_passages = relevant_passages + irrelevant_passages
        if self.image_kb is None:
            all_passages_text = all_passages
        else:
            all_passages_text = [p[self.kb_input_key] for p in all_passages]
        context_inputs_text = self.tokenizer(all_passages_text, **self.tokenization_kwargs)
        if self.passage_type_ids:
            context_inputs_text['token_type_ids'][context_inputs_text['attention_mask'].bool()] = 1
        
        # wrap it up
        question_inputs = self.image_formatter.format_batch(question_inputs_text, items)
        context_inputs = self.image_formatter.format_batch(context_inputs_text, all_passages)
        context_inputs['titles'] = all_titles
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch
        
       
class ReRankerDataModule(QuestionAnsweringDataModule):    
    """
    Parameters
    ----------
    *args, **kwargs: 
        additional arguments are passed to QuestionAnsweringDataModule
    run_path: str, optional
        Path to the ranx run stored in the TREC format that holds the IR results.
        Optional if you want to train only.
        Defaults to None.
    qrels_path: str, optional
        Path to the ranx qrels stored in the TREC format. Used during evaluation.
        Optional if you want to train only.
        Defaults to None.
    """
    def __init__(self, *args, run_path=None, qrels_path=None, **kwargs):
        super().__init__(*args, **kwargs)       
        if run_path is not None:
            self.run = ranx.Run.from_file(run_path)
        else:            
            self.run = None         
        if qrels_path is not None:
            self.qrels = ranx.Qrels.from_file(qrels_path)
        else:            
            self.qrels = None
            
    def get_eval_passages(self, item):
        """Keep the top-M passages retrieved by the IR"""
        ir_results = self.run.run[item['id']]
        if not ir_results:
            return []
        
        # document ids in ranx are str so we map them back to indices (int)
        indices = list(map(int, ir_results.keys()))[: self.M]
            
        passages = self.kb.select(indices)
        
        # multimodal vs. text-only mode
        if self.image_kb is None:
            passages = passages[self.kb_input_key]
        else:
            passages = self.add_image(list(passages))
        return passages
    
    def collate_fn(self, items):
        """
        Collate batch so that each question is associate with 1 and M-1 irrelevant ones.
        Also tokenizes input strings

        Returns (a dict of)
        -------------------
        input_ids: Tensor[int]
            shape (N * M, L)     
            1 relevant passage followed by M-1 irrelevant ones, N times
            /!\ different from BiEncoderDataModule
        labels: torch.LongTensor, optional
            shape (N, )
            Index of the relevant passage in input_ids.
            Should be 0 except for padding passages.
            Returned only when training.
        **kwargs: more tensors depending on the tokenizer, e.g. attention_mask
        """
        assert self.n_relevant_passages == 1
        question_ids, questions, passages, labels = [], [], [], []
        for item in items:
            questions.extend([item]*self.M)                            
            if self.trainer.state.stage != "train":
                passage = self.get_eval_passages(item)
                question_ids.extend([item['id']]*self.M)
            else:
                relevant_passage, irrelevant_passage = self.get_training_passages(item)
                passage = relevant_passage + irrelevant_passage
                if len(relevant_passage) < 1:
                    labels.append(self.trainer.lightning_module.loss_fct.ignore_index)
                else:
                    labels.append(0)

            passages.extend(passage)
            if len(passage) < self.M:
                passages.extend(self.padding_passage*(self.M-len(passage)))

        if self.image_kb is None:
            passages_text = passages
        else:
            passages_text = [p[self.kb_input_key] for p in passages]
        questions_text = [q[self.input_key] for q in questions]
        batch = self.tokenizer(*(questions_text, passages_text), **self.tokenization_kwargs)
        batch = self.image_formatter.format_batch(batch, questions, passages)
        if labels:
            batch['labels'] = torch.tensor(labels)
        if question_ids:
            batch['ids'] = question_ids
        return batch
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep question identifiers in batch. Does not try to cast them as Tensor of any dtype or device."""
        question_ids = batch.pop('ids', None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        batch['ids'] = question_ids
        return batch
    
    
def map_run(run, mapping, k=100):
    new_run={}
    for q_id, results in run.run.items():
        new_results = {}
        for doc_id, score in results.items():
            for i in mapping[doc_id]:
                new_results[str(i)] = score
            if len(new_results) >= k:
                break
        new_run[q_id] = new_results
    return ranx.Run(new_run)


class ReaderDataModule(QuestionAnsweringDataModule):
    """
    Parameters
    ----------
    *args, **kwargs: 
        additional arguments are passed to QuestionAnsweringDataModule
    max_n_answers: int, optional
        The answer might be found several time in the same passage, this is a threshold to enable batching
        Defaults to 10.
    train_original_answer_only: bool, optional
        Whether the model should be trained to predict only the original answer (default)
        or all alternative answers (with the only limit of max_n_answers)
        This has no effect on the evaluation (where all alternative answers are always considered)
    oracle: bool, optional
        Whether to use only relevant passages at inference (stored in {search_key}_provenance_indices)
        Will enforce n_relevant_passages=M
        Defaults to False (use IR passages at inference, stored in {search_key}_indices)
    run_path: str, optional
        Path to the ranx run stored in the TREC format that holds the IR results.
        To be used instead of search_key at inference.
        Defaults to None.
    extract_name: bool, optional
        Train the model to extract the name of the entity instead of the answer.
        Defaults to False.
    mapping_run: str, optional
        Path to the mapping
    """
    def __init__(self, *args, max_n_answers=10, run_path=None, 
                 train_original_answer_only=True, oracle=False, extract_name=False, 
                 mapping_run=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_n_answers = max_n_answers
        self.train_original_answer_only = train_original_answer_only
        self.oracle = oracle
        self.extract_name = extract_name
        
        if self.oracle and self.n_relevant_passages != self.M:
            warnings.warn(f"Oracle mode. Setting n_relevant_passages={self.M}")
            self.n_relevant_passages = self.M
        if run_path is not None:
            self.run = ranx.Run.from_file(run_path)
            if mapping_run is not None:
                with open(mapping_run, 'rt') as file:
                    mapping_run = json.load(file)
                self.run = map_run(self.run, mapping_run, k=self.M)
        else:            
            self.run = None
            
    def get_eval_passages(self, item):
        """Keep the top-M passages retrieved by the IR"""
        if self.run is None:
            indices = item[self.search_key+"_indices"][: self.M]
            scores = item[self.search_key+"_scores"][: self.M]
        else:
            ir_results = self.run.run[item['id']]
            if not ir_results:
                return [], []
            # document ids in ranx are str so we map them back to indices (int)
            indices = list(map(int, ir_results.keys()))[: self.M]
            scores = list(ir_results.values())[: self.M]
            
        passages = self.kb.select(indices)
        
        # multimodal vs. text-only mode
        if self.image_kb is None:
            passages = passages[self.kb_input_key]
        else:
            passages = self.add_image(list(passages))
        return passages, scores
                       
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
        return dict(start_positions=start_positions, end_positions=end_positions, answer_mask=answer_mask)
    
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
        with_scores = self.trainer.lightning_module.model.fuse_ir_score
        N = len(items)
        answer_mask = torch.zeros((N*self.M, self.max_n_answers), dtype=torch.long)
        for i, item in enumerate(items):
            # N. B. seed is set in Trainer
            questions.extend([item]*self.M)

            # oracle -> use only relevant passages
            if (self.trainer.state.stage != "train") and not self.oracle:
                passage, score = self.get_eval_passages(item)
                if len(score) < self.M:
                    score.extend([0]*(self.M-len(score)))
                passage_scores.append(score)
            else:
                relevant_passage, irrelevant_passage, *scores = self.get_training_passages(item, with_scores=with_scores)
                passage = relevant_passage + irrelevant_passage
                if with_scores:
                    relevant_scores, irrelevant_scores = scores
                    passage_scores.append(relevant_scores)
                    passage_scores.append(irrelevant_scores)
                    if (len(relevant_scores) + len(irrelevant_scores)) < self.M:
                        passage_scores.append(np.zeros(self.M-(len(relevant_scores) + len(irrelevant_scores)), dtype=np.float32))

            passages.extend(passage)
            # all passages have at least 1 non-masked answer (set to 0 for irrelevant passages)
            answer_mask[i*self.M: i*self.M+len(passage), 0] = 1
            # except for padding passages
            if len(passage) < self.M:
                passages.extend(self.padding_passage*(self.M-len(passage)))
            
            if self.extract_name:
                original_answer = item['wikidata_label']
                # FIXME: maybe train on aliases of the entity?
                answer = [original_answer]
                answer_strings.extend([answer]*self.M)
            else:
                original_answer = item['output']['original_answer']
                answer = item['output']['answer']
                # beware this create a discrepancy between answer_strings and answers (tokens)
                # evaluation should always be done using answer_strings
                if QuestionType[item.get('question_type', 'String')] == QuestionType.Numerical:
                    if self.image_kb is None:
                        passages_text = passage
                    else:
                        passages_text = [p[self.kb_input_key] for p in passage]
                    answer = find_valid_numerical_answers(answer, passages_text)
                    answer = answer if answer else ['']
                    answer_strings.extend([answer]*self.M)
                elif self.train_original_answer_only:
                    answer_strings.extend([answer]*self.M)
                    answer = [original_answer]            
                else:                
                    answer_strings.extend([answer]*self.M)
                    # avoid processing the same answer twice
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
        if self.image_kb is None:
            passages_text = passages
        else:
            passages_text = [p[self.kb_input_key] for p in passages]
        questions_text = [q[self.input_key] for q in questions]
        batch = self.tokenizer(*(questions_text, passages_text), **self.tokenization_kwargs)
        answer_position = self.get_answer_position(batch, answers, answer_mask)   
        batch = self.image_formatter.format_batch(batch, questions, passages)
        batch.update(answer_position)
        batch['answer_strings'] = answer_strings
        if passage_scores:
            batch['passage_scores'] = torch.tensor(np.concatenate(passage_scores, dtype=np.float32))

        return batch
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep answer_strings in batch. Does not try to cast them as Tensor of any dtype or device."""
        answer_strings = batch.pop('answer_strings', None)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        batch['answer_strings'] = answer_strings
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
    biencoder: bool, optional
        Expected kind of model: bi-encoder or cross-encoder
        i.e. whether to concatenate questions with passages or leave them in separate tensors
    sentences_per_target: int, optional
        Number of sentences in the target passages
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
    def __init__(self, *args, biencoder=True, sentences_per_target=4,
                 prepend_title=False, text_mask_rate=1.0, image_mask_rate=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_relevant_passages == 1
        if biencoder:
             self.collate_fn = self.biencoder_collate_fn
        else:
            self.collate_fn = self.reranker_collate_fn
        self.sentences_per_target = sentences_per_target
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
            
        # rename context image features/image path and copy for query
        if self.image_formatter.precomputed:
            for k in ({"face_box", "face_embedding"} | self.image_formatter.image_features.image_embeddings_keys):
                target[k] = item.get(f"{context_image_key}{k}")
                query[k] = item.get(k)
        else:
            target['image'] = item[f"{context_image_key}image"]
            query['image'] = item['image']
        return query, target

    def biencoder_collate_fn(self, items):        
        questions, relevant_passages, labels = [], [], []
        for i, item in enumerate(items):
            query, relevant_passage = self.get_pseudo_question(item)
            labels.append(i)
            questions.append(query)
            relevant_passages.append(relevant_passage)

        question_inputs_text = self.tokenizer([q['text'] for q in questions], **self.tokenization_kwargs)
        context_inputs_text = self.tokenizer([p['text'] for p in relevant_passages], **self.tokenization_kwargs)
        # get image features or pixels, for both questions and passages
        question_inputs = self.image_formatter.format_batch(question_inputs_text, questions)
        context_inputs = self.image_formatter.format_batch(context_inputs_text, relevant_passages)
        
        n_hard_negatives = self.M - self.n_relevant_passages
        # make n_hard_negatives by shifting the images of relevant passages
        if n_hard_negatives > 0:
            if not self.image_formatter.precomputed:
                raise NotImplementedError()
            # duplicate relevant text
            for k, v in context_inputs["text_inputs"].items():
                context_inputs["text_inputs"][k] = torch.tile(v, (n_hard_negatives+1, 1))
            # shift relevant images
            for k, v in context_inputs['image_inputs'].items():
                shifted_input, shifted_mask = [v[self.input_key]], [v['attention_mask']]
                for shift in range(n_hard_negatives):
                    # shift along axis 0 (batch axis)
                    shifted_input.append(torch.roll(v[self.input_key], shift+1, 0))
                    shifted_mask.append(torch.roll(v['attention_mask'], shift+1, 0))
                # cat along axis 0 (batch axis)
                v[self.input_key] = torch.cat(shifted_input, 0)
                v['attention_mask'] = torch.cat(shifted_mask, 0)
            # shift relevant faces
            shifted_faces, shifted_boxes = [context_inputs['face_inputs']["face"]], [context_inputs['face_inputs']["bbox"]]
            shifted_mask = [context_inputs['face_inputs']['attention_mask']]
            for shift in range(n_hard_negatives):
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
        
    def reranker_collate_fn(self, items):   
        assert len(items) >= self.M, "Not enough random negatives"
        # generate pseudo-questions for all items in the batch
        unique_questions, relevant_passages = [], []
        for item in items:
            query, relevant_passage = self.get_pseudo_question(item)
            unique_questions.append(query)
            relevant_passages.append(relevant_passage)
        
        # mix questions with random negatives (passages relevant for other questions in the batch)
        questions, passages = [], []
        for i in range(len(items)):
            for j in range(self.M):
                questions.append(unique_questions[i])
                # j==0 --> relevant passage. label should always be 0
                # j>0  --> irrelevant passage. relevant for another question in the batch
                if i+j < len(items):
                    passages.append(relevant_passages[i+j])
                # corner-case: get passages from the first questions in the batch
                else:
                    passages.append(relevant_passages[i+j-len(items)])
                    
        questions_text = [q['text'] for q in questions]
        passages_text = [p['text'] for p in passages]
        batch = self.tokenizer(*(questions_text, passages_text), **self.tokenization_kwargs)
        batch = self.image_formatter.format_batch(batch, questions, passages)

        # wrap it up
        batch['labels'] = torch.zeros(len(items), dtype=torch.long)
        return batch
