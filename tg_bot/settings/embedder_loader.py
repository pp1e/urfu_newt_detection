import os
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch import nn

from bot.belly_vectorization.build_transform import build_transforms
from bot.belly_vectorization.classifier_embedder import ResNetClassifierEmbedder

load_dotenv()

DEVICE = os.getenv("DEVICE", "cpu")
MODEL_EMBEDDER_PATH = os.getenv("MODEL_EMBEDDER_PATH")


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
