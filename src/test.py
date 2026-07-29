import argparse
import os
import random

import torch

from config import get_parser
from data import build_dataloaders
from metrics import calculate_roc_metrics
from model import build_model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def create_verification_pairs(embeddings, labels, number_of_pairs, seed):
    rng = random.Random(seed)
    label_to_indices = {}

    for index, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(label, []).append(index)

    unique_labels = list(label_to_indices.keys())
    pair_labels = []
    pair_distances = []

    for pair_index in range(number_of_pairs):
        same_person = pair_index % 2 == 0

        if same_person:
            label = rng.choice(unique_labels)
            first_index, second_index = rng.sample(
                label_to_indices[label],
                2,
            )
            pair_label = 1
        else:
            first_label, second_label = rng.sample(unique_labels, 2)
            first_index = rng.choice(label_to_indices[first_label])
            second_index = rng.choice(label_to_indices[second_label])
            pair_label = 0

        distance = torch.linalg.vector_norm(
            embeddings[first_index] - embeddings[second_index],
            ord=2,
        ).item()
        pair_labels.append(pair_label)
        pair_distances.append(distance)

    return pair_labels, pair_distances


def main():
    cli_cfg = get_parser()
    device = get_device()
    checkpoint_path = os.path.join(
        "checkpoints",
        f"{cli_cfg.experiment_name}_best.pth",
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_cfg = argparse.Namespace(**checkpoint["config"])
    model_cfg.dataset_root = cli_cfg.dataset_root
    model_cfg.batch_size = cli_cfg.batch_size
    model_cfg.num_workers = cli_cfg.num_workers
    model_cfg.identities_per_batch = cli_cfg.identities_per_batch
    model_cfg.images_per_identity = cli_cfg.images_per_identity
    model_cfg.validation_identity_ratio = cli_cfg.validation_identity_ratio

    loaders, _, _ = build_dataloaders(model_cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings, labels = extract_embeddings(
        model,
        loaders["test_images"],
        device,
    )
    pair_labels, pair_distances = create_verification_pairs(
        embeddings,
        labels,
        number_of_pairs=10000,
        seed=model_cfg.seed,
    )
    results = calculate_roc_metrics(pair_labels, pair_distances)

    positive_distances = [
        distance
        for label, distance in zip(pair_labels, pair_distances)
        if label == 1
    ]
    negative_distances = [
        distance
        for label, distance in zip(pair_labels, pair_distances)
        if label == 0
    ]

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test identities: {len(set(labels.tolist()))}")
    print(f"Test images: {len(labels)}")
    print(
        "Mean positive distance: "
        f"{sum(positive_distances) / len(positive_distances):.4f}"
    )
    print(
        "Mean negative distance: "
        f"{sum(negative_distances) / len(negative_distances):.4f}"
    )
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"EER: {results['eer']:.2%}")
    print(f"EER distance threshold: {-results['eer_threshold']:.4f}")


if __name__ == "__main__":
    main()
