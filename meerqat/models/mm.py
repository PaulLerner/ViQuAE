from torch import nn


class IntermediateLinearFusion(nn.Module):
    """Fuses DPR’s text representation with image embeddings by projecting them linearly in the same space"""
    def __init__(self, dpr_encoder, face_embedding, image_embeddings, embedding_dim, dropout=0.1, layer_norm_eps=1e-12):
        """
        Arguments
        ---------
        dpr_encoder: DPRContextEncoder or DPRQuestionEncoder
        face_embedding: FaceEmbedding
        image_embeddings: nn.ModuleDict[ImageEmbedding]
        """
        super().__init__()
        self.dpr_encoder = dpr_encoder
        self.face_embedding = face_embedding
        self.image_embeddings = image_embeddings
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
        for face in face_inputs:
            output += self.face_embedding(**face)
        for name, image in image_inputs.items():
            output += self.image_embeddings[name](**image)
        output = self.LayerNorm(output)
        return self.dropout(output)
