import torch
from torch import nn
from torchvision import models


class ResNetClassifierEmbedder(nn.Module):
    """
    ResNet -> embedding -> classifier head (for training).
    For inference, use forward_embedding().
    """
    def __init__(self, num_classes: int, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.embedding(feats)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embedding(x)
        logits = self.classifier(emb)
        return logits
