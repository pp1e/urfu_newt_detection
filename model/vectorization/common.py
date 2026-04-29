import random
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import umap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tg_bot.bot.belly_vectorization.classifier_embedder import DinoViTClassifierEmbedder


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


MAX_CMC_RANK = 10


def save_cmc_curve(
    cmc_curve: torch.Tensor,
    save_path: Path,
    cmc_std: torch.Tensor | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ranks = torch.arange(1, len(cmc_curve) + 1).cpu().numpy()
    values = cmc_curve.cpu().numpy()

    ranks = ranks[:MAX_CMC_RANK]
    values = values[:MAX_CMC_RANK]

    if cmc_std is not None:
        std_values = cmc_std.cpu().numpy()[:MAX_CMC_RANK]
        lower = values - std_values
        upper = values + std_values

        lower = lower.clip(min=0.0)
        upper = upper.clip(max=1.0)

        plt.fill_between(ranks, lower, upper, alpha=0.2, label="± std")

    plt.figure(figsize=(8, 5))
    plt.plot(ranks, values, marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Identification rate")
    plt.title("CMC curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_plot_umap(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    max_classes: int=15,
):
    """
    embeddings: Tensor [N, D]
    labels: Tensor [N]
    """

    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # ограничим число классов (чтобы не было каши)
    unique_classes = np.unique(labels)
    selected_classes = unique_classes[:max_classes]

    mask = np.isin(labels, selected_classes)

    embeddings = embeddings[mask]
    labels = labels[mask]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    embeddings_2d = reducer.fit_transform(embeddings)

    cmap = plt.cm.get_cmap("tab20", len(unique_classes))

    plt.figure(figsize=(8, 6))

    for i, class_id in enumerate(np.unique(labels)):
        idx = labels == class_id
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            color=cmap(i),
            label=f"class {class_id}",
            s=20,
        )

    plt.title("UMAP визуализация по особям")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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

    for x, y, _ in tqdm(loader, desc="Extract embeddings", unit="batch"):
        x = x.to(device, non_blocking=True)

        emb = model.forward_embedding(x)
        emb = F.normalize(emb, dim=1)

        all_embeddings.append(emb.cpu())
        all_labels.append(y.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def calculate_cmc_curve(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> torch.Tensor:
    similarities = query_embeddings @ gallery_embeddings.T

    num_queries = query_embeddings.shape[0]
    gallery_size = gallery_embeddings.shape[0]

    cmc_hits = torch.zeros(gallery_size, dtype=torch.float32)

    for query_index in range(num_queries):
        similarities_for_query = similarities[query_index]
        order = torch.argsort(similarities_for_query, descending=True)

        sorted_gallery_labels = gallery_labels[order]
        matches = (sorted_gallery_labels == query_labels[query_index]).to(torch.int64)

        relevant_positions = torch.nonzero(matches, as_tuple=False).flatten()

        if relevant_positions.numel() == 0:
            continue

        first_hit_rank = int(relevant_positions[0].item())
        cmc_hits[first_hit_rank:] += 1.0

    return cmc_hits / max(1, num_queries)

