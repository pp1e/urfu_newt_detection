from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.vectorization.datasets import build_datasets
from tg_bot.bot.belly_vectorization.constants import IMAGE_SIZE
from tg_bot.bot.belly_vectorization.classifier_embedder import DinoViTClassifierEmbedder
from tg_bot.bot.belly_vectorization.build_transform import build_transforms


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    embedding_dim = checkpoint["embedding_dim"]
    class_to_id = checkpoint["class_to_id"]
    num_classes = len(class_to_id)

    model = DinoViTClassifierEmbedder(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    ).to(device)

    # model = ResNetClassifierEmbedder(
    #     num_classes=num_classes,
    #     embedding_dim=embedding_dim,
    # ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []

    for x, y, paths in tqdm(loader, desc="Extract embeddings", unit="batch"):
        x = x.to(device, non_blocking=True)

        emb = model.forward_embedding(x)
        emb = F.normalize(emb, dim=1)

        all_embeddings.append(emb.cpu())
        all_labels.append(y.cpu())
        all_paths.extend(paths)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels, all_paths


def average_precision_at_hits(sorted_relevance: torch.Tensor) -> float:
    """
    sorted_relevance: tensor shape [N], 1 если элемент релевантен, иначе 0
    """
    total_relevant = int(sorted_relevance.sum().item())
    if total_relevant == 0:
        return 0.0

    cumsum = torch.cumsum(sorted_relevance, dim=0)
    positions = torch.arange(1, len(sorted_relevance) + 1, dtype=torch.float32)
    precision_at_k = cumsum.float() / positions

    ap = (precision_at_k * sorted_relevance.float()).sum() / total_relevant
    return float(ap.item())


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
):
    similarities = query_embeddings @ gallery_embeddings.T

    num_queries = query_embeddings.shape[0]
    rank1_hits = 0
    rank5_hits = 0
    ap_values: list[float] = []

    for i in range(num_queries):
        sims = similarities[i]
        order = torch.argsort(sims, descending=True)

        sorted_gallery_labels = gallery_labels[order]
        matches = (sorted_gallery_labels == query_labels[i]).to(torch.int64)

        if len(matches) > 0 and matches[0].item() == 1:
            rank1_hits += 1

        topk = min(5, len(matches))
        if matches[:topk].sum().item() > 0:
            rank5_hits += 1

        ap = average_precision_at_hits(matches)
        ap_values.append(ap)

    rank1 = rank1_hits / max(1, num_queries)
    rank5 = rank5_hits / max(1, num_queries)
    mean_ap = sum(ap_values) / max(1, len(ap_values))

    return {
        "rank1": rank1,
        "rank5": rank5,
        "mAP": mean_ap,
        "num_queries": num_queries,
        "gallery_size": int(gallery_embeddings.shape[0]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    _, val_tf = build_transforms(image_size)

    datasets = build_datasets(
        data_root=args.data_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        val_tf=val_tf,
        train_tf=val_tf,
        is_evaluate=True,
    )

    model, _ = build_model(args.checkpoint, device)

    gallery_embeddings, gallery_labels, _ = extract_embeddings(
        model=model,
        dataset=datasets.train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    query_embeddings, query_labels, _ = extract_embeddings(
        model=model,
        dataset=datasets.val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    metrics = evaluate_retrieval(
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
    )

    print("\nRetrieval metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
