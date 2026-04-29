from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.vectorization.common import save_cmc_curve, extract_embeddings, calculate_cmc_curve, build_model, seed_all
from model.vectorization.datasets import build_evaluation_datasets
from tg_bot.bot.belly_vectorization.constants import IMAGE_SIZE
from tg_bot.bot.belly_vectorization.build_transform import build_transforms


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


@dataclass
class EvaluationMetrics:
    rank1: float
    rank5: float
    mean_ap: float
    num_queries: int
    gallery_size: int
    cmc_curve: Tensor


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> EvaluationMetrics:
    similarities = query_embeddings @ gallery_embeddings.T

    gallery_size = int(gallery_embeddings.shape[0])

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


    return EvaluationMetrics(
        cmc_curve = calculate_cmc_curve(
            query_embeddings=query_embeddings,
            query_labels=query_labels,
            gallery_embeddings=gallery_embeddings,
            gallery_labels=gallery_labels,
        ),
        rank1 = rank1_hits / max(1, num_queries),
        rank5 = rank5_hits / max(1, num_queries),
        mean_ap = sum(ap_values) / max(1, len(ap_values)),
        num_queries = num_queries,
        gallery_size = gallery_size,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cmc-save-path", type=Path, default=Path("cmc-curve.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = checkpoint.get("image_size", IMAGE_SIZE)

    _, val_tf = build_transforms(image_size)

    datasets = build_evaluation_datasets(
        data_root=args.data_root,
        seed=args.seed,
        transform=val_tf,
    )

    model, _ = build_model(args.checkpoint, device)

    gallery_embeddings, gallery_labels = extract_embeddings(
        model=model,
        dataset=datasets.gallery,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    query_embeddings, query_labels = extract_embeddings(
        model=model,
        dataset=datasets.query,
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

    save_cmc_curve(
        cmc_curve=metrics.cmc_curve,
        save_path=Path(args.cmc_save_path),
    )

    print(f"CMC curve saved to {args.cmc_save_path}")

    print("\nRetrieval metrics:")
    print(f"Rank 1: {metrics.rank1:.4f}")
    print(f"Rank 5: {metrics.rank5:.4f}")
    print(f"Mean AP: {metrics.mean_ap:.4f}")
    print(f"Query size: {metrics.num_queries}")
    print(f"Gallery size: {metrics.gallery_size}")
    print(f"Classes number: {len(datasets.class_to_id)}")


if __name__ == "__main__":
    main()
