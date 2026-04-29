from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.vectorization.common import save_cmc_curve, extract_embeddings, calculate_cmc_curve, MAX_CMC_RANK
from model.vectorization.datasets import build_evaluation_datasets
from tg_bot.bot.belly_vectorization.build_transform import build_transforms
from tg_bot.bot.belly_vectorization.classifier_embedder import DinoViTClassifierEmbedder
from tg_bot.bot.belly_vectorization.constants import IMAGE_SIZE


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

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint


def evaluate_cmc_for_run(
    *,
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tensor:
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Dataset:    {data_root}")

    model, checkpoint = build_model(checkpoint_path, device)

    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    _, val_transform = build_transforms(image_size)

    datasets = build_evaluation_datasets(
        data_root=data_root,
        transform=val_transform,
        seed=seed,
    )

    print(f"Identities: {len(datasets.class_to_id)}")
    print(f"Gallery:    {len(datasets.gallery)}")
    print(f"Query:      {len(datasets.query)}")

    gallery_embeddings, gallery_labels = extract_embeddings(
        model=model,
        dataset=datasets.gallery,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    query_embeddings, query_labels = extract_embeddings(
        model=model,
        dataset=datasets.query,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    return calculate_cmc_curve(
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
    )


def build_mean_cmc(cmc_curves: list[Tensor]) -> tuple[Tensor, Tensor]:
    if not cmc_curves:
        raise RuntimeError("No CMC curves were provided")

    min_length = min(len(cmc_curve) for cmc_curve in cmc_curves)
    aligned_curves = [cmc_curve[:min_length] for cmc_curve in cmc_curves]

    cmc_stack = torch.stack(aligned_curves, dim=0)

    cmc_mean = cmc_stack.mean(dim=0)
    cmc_std = cmc_stack.std(dim=0)

    print(cmc_mean)
    print(cmc_std)

    return cmc_mean, cmc_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        required=True,
        help="Directory with experiments-embedder-1.pt ... experiments-embedder-5.pt",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="Directory with sample1/experimental_eval ... sample5/experimental_eval",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Run numbers. Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("mean-cmc-curve.png"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    cmc_curves: list[Tensor] = []

    for run_number in args.runs:
        checkpoint_path = (
            args.checkpoints_root
            / f"experiments-embedder-{run_number}.pt"
        )

        data_root = (
            args.datasets_root
            # / "additional-bellies"
            / f"sample{run_number}"
            / "experimental_eval"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if not data_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_root}")

        cmc_curve = evaluate_cmc_for_run(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        cmc_curves.append(cmc_curve)

    cmc_mean, cmc_std = build_mean_cmc(cmc_curves)

    save_cmc_curve(
        cmc_curve=cmc_mean,
        cmc_std=cmc_std,
        save_path=args.output_path,
    )

    print(f"\nMean CMC curve saved to: {args.output_path}")

    print("\nMean CMC values:")
    for rank, value in enumerate(cmc_mean[: MAX_CMC_RANK], start=1):
        print(f"Rank-{rank}: {value:.4f}")


if __name__ == "__main__":
    main()