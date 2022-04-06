from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput

from meerqat.models.image import ImageEmbedding, FaceEmbedding


@dataclass 
class EncoderOutput(ModelOutput):
    pooler_output: Optional[torch.FloatTensor] = None


class DMREncoder(nn.Module):
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

    def __init__(
            self, bert_model, embedding_dim=None,
            dropout=0.1, layer_norm_eps=1e-12, face_kwargs={}, image_kwargs={}
    ):
        """
        Arguments
        ---------
        bert_model: BertModel
        """
        super().__init__()
        self.bert_model = bert_model
        if embedding_dim is None:
            embedding_dim = self.bert_model.config.hidden_size
        self.face_embedding = FaceEmbedding(embedding_dim=embedding_dim, dropout=dropout,
                                            layer_norm_eps=layer_norm_eps, **face_kwargs)
        self.image_embeddings = nn.ModuleDict()
        for name, image_kwarg in image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=embedding_dim, dropout=dropout,
                                                         **image_kwarg)

    def forward(self, text_inputs, face_inputs, image_inputs):
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
        faces = faces.reshape(batch_size * n_faces, face_dim)
        # embed batch of size batch_size*n_faces
        face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size * n_faces, -1))
        face_output = face_output.reshape(batch_size, n_faces, -1)

        # embed images
        image_outputs, image_attention_mask = [], []
        for name, image in image_inputs.items():
            image_outputs.append(self.image_embeddings[name](image['input']))
            image_attention_mask.append(image['attention_mask'])
        # (n_images, batch_size, embedding_dim) -> (batch_size, n_images, embedding_dim)
        image_outputs = torch.cat(image_outputs, 0).transpose(0, 1)
        image_attention_mask = torch.cat(image_attention_mask, 0).transpose(0, 1)

        # embed text: (batch_size, sequence_length, embedding_dim)
        text_embeddings = self.bert_model.embeddings(input_ids=text_inputs['input_ids'],
                                                     token_type_ids=text_inputs.get('token_type_ids'))

        # (batch_size, sequence_length+n_faces+n_images, embedding_dim)
        multimodal_embeddings = torch.cat((text_embeddings, face_output, image_outputs), dim=1)
        attention_mask = torch.cat((text_inputs['attention_mask'], face_inputs['attention_mask'], image_attention_mask), dim=1)

        outputs = self.bert_model.encoder(multimodal_embeddings, attention_mask=attention_mask)

        # same as DPR: extract representation from [CLS]: the first token
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        return EncoderOutput(pooler_output=pooled_output)


class IntermediateLinearFusion(nn.Module):
    """Fuses DPR’s text representation with image embeddings by projecting them linearly in the same space"""
    def __init__(
            self, dpr_encoder, embedding_dim=None, 
            dropout=0.1, layer_norm_eps=1e-12, face_kwargs={}, image_kwargs={}
        ):
        """
        Arguments
        ---------
        dpr_encoder: DPRContextEncoder or DPRQuestionEncoder
        """
        super().__init__()
        self.dpr_encoder = dpr_encoder
        if embedding_dim is None:
            embedding_dim = self.dpr_encoder.config.hidden_size
        self.face_embedding = FaceEmbedding(embedding_dim=embedding_dim, dropout=dropout, 
                                            layer_norm_eps=layer_norm_eps, **face_kwargs)
        self.image_embeddings = nn.ModuleDict()
        for name, image_kwarg in image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=embedding_dim, dropout=dropout, **image_kwarg)
        self.dpr_proj = nn.Linear(dpr_encoder.config.hidden_size, embedding_dim)
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

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
        faces = faces.reshape(batch_size*n_faces, face_dim)
        # embed batch of size batch_size*n_faces
        face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size*n_faces, -1))
        face_output = face_output.reshape(batch_size, n_faces, -1)
        # sum over all faces
        face_output = face_output.sum(axis=1)

        # embed text
        output = self.dpr_encoder(**text_inputs).pooler_output
        output = self.dpr_proj(output)

        # fuse text and image
        output += face_output
        for name, image in image_inputs.items():
            output += self.image_embeddings[name](image['input'])
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return EncoderOutput(pooler_output=output)
