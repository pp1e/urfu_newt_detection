import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn
from torchvision import models
from torchvision.transforms import v2 as T


load_dotenv()

DEVICE = os.getenv("DEVICE", "cpu")
MODEL_EMBEDDER_PATH = os.getenv("MODEL_EMBEDDER_PATH")


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


def build_transforms(image_size: int = 224):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.Resize((image_size, image_size)),

        # брюшко может быть вверх ногами -> 50% переворачиваем на 180
        T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.5),

        # лёгкие фотометрические аугментации
        T.RandomApply(
            [T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)],
            p=0.7
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),

        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


@dataclass(frozen=True)
class Embedder:
    model: ResNetClassifierEmbedder
    transform: nn.Module
    device: torch.device
    embedding_dim: int


def build_embedder_model(
    num_classes: int,
    embedding_dim: int = 256,
    pretrained: bool = True,
):
    return ResNetClassifierEmbedder(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
    )


def load_embedder(checkpoint_path: Path, device: str = "cpu") -> Embedder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    class_to_id = ckpt.get("class_to_id")
    if not isinstance(class_to_id, dict) or not class_to_id:
        raise RuntimeError("Checkpoint must contain class_to_id (use training checkpoint).")

    embedding_dim = int(ckpt.get("embedding_dim", 256))
    image_size = int(ckpt.get("image_size", 224))
    num_classes = len(class_to_id)

    torch_device = torch.device(device)

    model = ResNetClassifierEmbedder(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(torch_device)
    model.eval()

    _, val_tf = build_transforms(image_size)

    return Embedder(
        model=model,
        transform=val_tf,
        device=torch_device,
        embedding_dim=embedding_dim,
    )


EMBEDDER_MODEL = load_embedder(
    Path(MODEL_EMBEDDER_PATH), device=DEVICE,
)
