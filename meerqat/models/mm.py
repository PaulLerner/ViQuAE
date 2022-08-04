from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers import PreTrainedModel, BertModel, DPRQuestionEncoder, DPRContextEncoder
from transformers.models.bert import BertConfig

from meerqat.models.image import ImageEmbedding, FaceEmbedding


@dataclass
class EncoderOutput(ModelOutput):
    pooler_output: Optional[torch.FloatTensor] = None


@dataclass 
class DMREncoderOutput(EncoderOutput):
    """
    Same as DPRQuestionEncoderOutput / DPRContextEncoderOutput
    """
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MMConfig(BertConfig):
    def __init__(self,
                 *args,
                 n_faces=4,
                 face_kwargs=None,
                 image_kwargs=None,
                 face_and_image_are_exclusive=False,
                 no_text=False,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.n_faces = n_faces
        if face_kwargs is None:
            self.face_kwargs = dict(face_dim=512, bbox_dim=7)
        else:
            self.face_kwargs = face_kwargs
        if image_kwargs is None:
            self.image_kwargs = {
                "clip-RN50": {"input_dim": 1024},
                "imagenet-RN50": {"input_dim": 2048}
            }
        else:
            self.image_kwargs = image_kwargs
        self.face_and_image_are_exclusive = face_and_image_are_exclusive
        self.no_text = no_text


class DMREncoder(PreTrainedModel):
    """
    Text and image are fused by concatenating them at the sequence-level then feeding them to BERT, à la UNITER (Chen et al.)
      one face ≃ one token
      one image ≃ one token

    The multimodal representation is obtained from the "[CLS]" token

    References
    ----------
    @inproceedings{chen_uniter_2020,
        title = {{UNITER}: {UNiversal} {Image}-{TExt} {Representation} {Learning}},
        url = {https://openreview.net/forum?id=S1eL4kBYwr},
        booktitle = {{ECCV} 2020},
        author = {Chen, Yen-Chun and Li, Linjie and Yu, Licheng and Kholy, Ahmed El and Ahmed, Faisal and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
        year = {2020},
        note = {https://github.com/ChenRocks/UNITER}
    }
    """
    config_class = MMConfig
    load_tf_weights = None
    base_model_prefix = "bert_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # FIXME: set add_pooling_layer=False
        self.bert_model = BertModel(config)
        if self.config.n_faces > 0:
            self.face_embedding = FaceEmbedding(embedding_dim=self.config.hidden_size,
                                                dropout=self.config.hidden_dropout_prob,
                                                layer_norm_eps=self.config.layer_norm_eps,
                                                **self.config.face_kwargs)
        else:
            self.face_embedding = None
        self.image_embeddings = nn.ModuleDict()
        for name, image_kwarg in self.config.image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=self.config.hidden_size,
                                                         dropout=self.config.hidden_dropout_prob,
                                                         **image_kwarg)
            
    def _init_weights(self, module):
        # keep torch defaults
        pass
    
    def forward(self, text_inputs, face_inputs, image_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        """
        Arguments
        ---------
        text_inputs: dict[str, torch.LongTensor]
            usual BERT inputs, see transformers.BertModel
        face_inputs: dict[str, torch.FloatTensor]
        {
            "face": (batch_size, n_faces, face_dim),
            "bbox": (batch_size, n_faces, bbox_dim),
            "attention_mask": (batch_size, n_faces)
        }
        image_inputs: dict[str, dict[str, torch.FloatTensor]]
        {
            model:
            {
                "input": (batch_size, image_dim)
                "attention_mask": (batch_size, )
            }
        }
        """
        # reshape faces
        faces = face_inputs['face']
        batch_size, n_faces, face_dim = faces.shape
        if n_faces > 0:
            faces = faces.reshape(batch_size * n_faces, face_dim)
            # embed batch of size batch_size*n_faces
            face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size * n_faces, -1))
            face_output = face_output.reshape(batch_size, n_faces, -1)
        else:
            face_output = torch.zeros(batch_size, 0, self.config.hidden_size, device=faces.device)

        # embed images
        image_outputs, image_attention_mask = [], []
        for name, image in image_inputs.items():
            image_outputs.append(self.image_embeddings[name](image['input']).unsqueeze(0))
            image_attention_mask.append(image['attention_mask'].unsqueeze(0))
        # (n_images, batch_size, embedding_dim) -> (batch_size, n_images, embedding_dim)
        image_outputs = torch.cat(image_outputs, 0).transpose(0, 1)
        image_attention_mask = torch.cat(image_attention_mask, 0).transpose(0, 1)
        
        if self.config.face_and_image_are_exclusive:
            face_attention_mask = face_inputs["attention_mask"]
            # indices at the batch level: at least one face detected (i.e. not masked)
            where_are_faces = face_attention_mask.nonzero()[:,0].unique()
            # mask images if at least one face was detected
            image_attention_mask[where_are_faces] = 0

        token_type_ids = text_inputs.get('token_type_ids')
        # keep only keep [CLS] token
        if self.config.no_text:
            text_inputs['input_ids'] = text_inputs['input_ids'][:, :1]
            text_inputs['attention_mask'] = text_inputs['attention_mask'][:, :1]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :1]
                
        # embed text: (batch_size, sequence_length, embedding_dim)
        text_embeddings = self.bert_model.embeddings(input_ids=text_inputs['input_ids'],
                                                     token_type_ids=token_type_ids)

        # (batch_size, sequence_length+n_faces+n_images, embedding_dim)
        multimodal_embeddings = torch.cat((text_embeddings, face_output, image_outputs), dim=1)
        attention_mask = torch.cat((text_inputs['attention_mask'], face_inputs['attention_mask'], image_attention_mask), dim=1)
        extended_attention_mask = self.bert_model.get_extended_attention_mask(
            attention_mask, multimodal_embeddings.shape[:-1], multimodal_embeddings.device
        )
        outputs = self.bert_model.encoder(multimodal_embeddings, attention_mask=extended_attention_mask,
                                          output_attentions=output_attentions,
                                          output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)

        # same as DPR: extract representation from [CLS]: the first token
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        
        if not return_dict:
            return (pooled_output, ) + outputs[2:]
        
        return DMREncoderOutput(
                pooler_output=pooled_output,
                hidden_states=outputs.hidden_states, 
                attentions=outputs.attentions)


class ILFConfig(MMConfig):
    def __init__(self,
                 *args,
                 question_encoder=True,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.question_encoder = question_encoder
         
            
class IntermediateLinearFusion(PreTrainedModel):
    """Fuses DPR’s text representation with image embeddings by projecting them linearly in the same space"""
    config_class = ILFConfig
    load_tf_weights = None
    base_model_prefix = "dpr_encoder"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.question_encoder:
            self.dpr_encoder = DPRQuestionEncoder(config)
        else:
            self.dpr_encoder = DPRContextEncoder(config)
        self.face_embedding = FaceEmbedding(embedding_dim=self.config.hidden_size, dropout=self.config.hidden_dropout_prob,
                                            layer_norm_eps=self.config.layer_norm_eps, **self.config.face_kwargs)
        self.image_embeddings = nn.ModuleDict()
        for name, image_kwarg in self.config.image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=self.config.hidden_size, dropout=self.config.hidden_dropout_prob, **image_kwarg)
        self.dpr_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
    def _init_weights(self, module):
        # keep torch defaults
        pass
    
    def forward(self, text_inputs, face_inputs, image_inputs):
        """
        Arguments
        ---------
        text_inputs: dict[str, torch.LongTensor]
            usual BERT inputs, see transformers.DPRQuestionEncoder
        face_inputs: dict[str, torch.FloatTensor]
        {
            "face": (batch_size, n_faces, face_dim),
            "bbox": (batch_size, n_faces, bbox_dim),
            "attention_mask": (batch_size, n_faces)
        }
        image_inputs: dict[str, dict[str, torch.FloatTensor]]
        {
            model:
            {
                "input": (batch_size, image_dim)
                "attention_mask": (batch_size, )
            }
        }
        """
        # reshape faces
        faces = face_inputs['face']
        batch_size, n_faces, face_dim = faces.shape
        faces = faces.reshape(batch_size * n_faces, face_dim)
        # embed batch of size batch_size*n_faces
        face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size * n_faces, -1))
        face_output = face_output.reshape(batch_size, n_faces, -1)
        # sum over all faces
        face_output = face_output.sum(axis=1)

        # embed text
        output = self.dpr_encoder(**text_inputs).pooler_output
        output = self.dpr_proj(output)

        # fuse text and image
        output += face_output
        if self.config.face_and_image_are_exclusive:
            face_attention_mask = face_inputs["attention_mask"]
            # indices at the batch level: at least one face detected (i.e. not masked)
            where_are_faces = face_attention_mask.nonzero()[:,0].unique()
        for name, image in image_inputs.items():
            # mask images if at least one face was detected
            if self.config.face_and_image_are_exclusive:
                image['input'][where_are_faces] = 0
            output += self.image_embeddings[name](image['input'])
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return EncoderOutput(pooler_output=output)
