from dataclasses import dataclass
from typing import Optional

from torch import nn
from transformers.modeling_outputs import ModelOutput

from meerqat.models.image import ImageEmbedding, FaceEmbedding


@dataclass 
class EncoderOutput(ModelOutput):
    pooler_output: Optional[torch.FloatTensor] = None


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
        text_inputs: dict[torch.LongTensor]
            usual BERT inputs, see transformers.DPRQuestionEncoder
        face_inputs: list[dict[torch.FloatTensor]]
            list of the same size as the number of faces detected in the images
        image_inputs: dict[dict[torch.FloatTensor]]
            {model: {"input": tensor}}
        """
        output = self.dpr_encoder(**text_inputs).pooler_output
        output = self.dpr_proj(output)
        for face in face_inputs:
            output += self.face_embedding(**face)
        for name, image in image_inputs.items():
            output += self.image_embeddings[name](**image)
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return EncoderOutput(pooler_output=output)
