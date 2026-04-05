import torch
from torch import nn

from .constants import IMAGE_SIZE


class DinoViTClassifierEmbedder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()

        import timm

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2",
            pretrained=pretrained,
            num_classes=0,
            img_size=IMAGE_SIZE,
        )

        backbone_dim = self.backbone.num_features

        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.GELU(),
            # nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.embedding_head(feats)
        emb = nn.functional.normalize(emb, dim=1)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_embedding(x)
        logits =  self.classifier(emb)
        return logits
