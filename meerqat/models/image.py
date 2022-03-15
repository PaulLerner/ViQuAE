from torch import nn


class FaceEmbedding(nn.Module):
    """Projects a face feature in the embedding space using a linear layer together with the corresponding bounding box."""
    def __init__(self, face_dim, bbox_dim, embedding_dim, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.face_proj = nn.Linear(face_dim, embedding_dim)
        self.bbox_proj = nn.Linear(bbox_dim, embedding_dim)
        self.LayerNorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, face, bbox):
        embedding = self.face_proj(face) + self.bbox_proj(bbox)
        embedding = self.LayerNorm(embedding)
        return self.dropout(embedding)


class ImageEmbedding(nn.Module):
    """Projects an image feature in the embedding space using a linear layer."""
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        embedding = self.linear(input)
        return self.dropout(embedding)
