import os
import random
import time

import torch
from torch import nn

from config import get_parser
from data import build_dataloaders
from model import build_model
from visualization import plot_training_history, save_history


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def calculate_triplet_metrics(anchor, positive, negative, margin):
    positive_distance = torch.linalg.vector_norm(
        anchor - positive,
        ord=2,
        dim=1,
    )
    negative_distance = torch.linalg.vector_norm(
        anchor - negative,
        ord=2,
        dim=1,
    )
    triplet_rate = (
        negative_distance > positive_distance + margin
    ).float().mean()
    return (
        positive_distance.mean(),
        negative_distance.mean(),
        triplet_rate,
    )


def mine_semi_hard_triplets(embeddings, labels, margin):
    with torch.no_grad():
        distances = torch.cdist(embeddings, embeddings, p=2)
        same_identity = labels[:, None] == labels[None, :]
        diagonal = torch.eye(
            labels.size(0),
            dtype=torch.bool,
            device=labels.device,
        )
        positive_mask = same_identity & ~diagonal
        negative_mask = ~same_identity

        positive_distances = distances.masked_fill(
            ~positive_mask,
            float("-inf"),
        )
        hardest_positive_indices = positive_distances.argmax(dim=1)
        hardest_positive_distances = distances.gather(
            1,
            hardest_positive_indices.unsqueeze(1),
        ).squeeze(1)

        semi_hard_mask = (
            negative_mask
            & (distances > hardest_positive_distances.unsqueeze(1))
            & (
                distances
                < hardest_positive_distances.unsqueeze(1) + margin
            )
        )
        semi_hard_distances = distances.masked_fill(
            ~semi_hard_mask,
            float("inf"),
        )
        semi_hard_negative_indices = semi_hard_distances.argmin(dim=1)
        has_semi_hard_negative = semi_hard_mask.any(dim=1)

        fallback_distances = distances.masked_fill(
            ~negative_mask,
            float("-inf"),
        )
        fallback_negative_indices = fallback_distances.argmax(dim=1)
        negative_indices = torch.where(
            has_semi_hard_negative,
            semi_hard_negative_indices,
            fallback_negative_indices,
        )

    return (
        embeddings,
        embeddings[hardest_positive_indices],
        embeddings[negative_indices],
    )


def train_one_epoch(model, data_loader, optimizer, criterion, device, margin):
    model.train()
    total_loss = 0.0
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_triplet_rate = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(images)
        anchor_embeddings, positive_embeddings, negative_embeddings = (
            mine_semi_hard_triplets(embeddings, labels, margin)
        )

        loss = criterion(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
        )
        positive_distance, negative_distance, triplet_rate = (
            calculate_triplet_metrics(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings,
                margin,
            )
        )
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_positive_distance += positive_distance.item() * batch_size
        total_negative_distance += negative_distance.item() * batch_size
        total_triplet_rate += triplet_rate.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "positive_distance": total_positive_distance / total_samples,
        "negative_distance": total_negative_distance / total_samples,
        "triplet_rate": total_triplet_rate / total_samples,
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, margin):
    model.eval()
    total_loss = 0.0
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_triplet_rate = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        anchor_embeddings, positive_embeddings, negative_embeddings = (
            mine_semi_hard_triplets(embeddings, labels, margin)
        )
        loss = criterion(
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
        )
        positive_distance, negative_distance, triplet_rate = (
            calculate_triplet_metrics(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings,
                margin,
            )
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_positive_distance += positive_distance.item() * batch_size
        total_negative_distance += negative_distance.item() * batch_size
        total_triplet_rate += triplet_rate.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "positive_distance": total_positive_distance / total_samples,
        "negative_distance": total_negative_distance / total_samples,
        "triplet_rate": total_triplet_rate / total_samples,
    }


def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    cfg = get_parser()
    set_seed(cfg.seed)
    device = get_device()
    loaders, class_names, split_class_names = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    criterion = nn.TripletMarginLoss(
        margin=cfg.triplet_margin,
        p=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = os.path.join(
        "checkpoints",
        f"{cfg.experiment_name}_best.pth",
    )
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_positive_distance": [],
        "val_positive_distance": [],
        "train_negative_distance": [],
        "val_negative_distance": [],
        "train_triplet_rate": [],
        "val_triplet_rate": [],
        "epoch_time": [],
    }

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            loaders["train_triplet"],
            optimizer,
            criterion,
            device,
            cfg.triplet_margin,
        )
        val_metrics = evaluate(
            model,
            loaders["val_triplet"],
            criterion,
            device,
            cfg.triplet_margin,
        )
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"PosDist={val_metrics['positive_distance']:.4f} "
            f"NegDist={val_metrics['negative_distance']:.4f} "
            f"TripletRate={val_metrics['triplet_rate']:.2%} "
            f"Time={format_time(epoch_time)}"
        )

        for split_name, metrics in (
            ("train", train_metrics),
            ("val", val_metrics),
        ):
            history[f"{split_name}_loss"].append(metrics["loss"])
            history[f"{split_name}_positive_distance"].append(
                metrics["positive_distance"]
            )
            history[f"{split_name}_negative_distance"].append(
                metrics["negative_distance"]
            )
            history[f"{split_name}_triplet_rate"].append(
                metrics["triplet_rate"]
            )
        history["epoch_time"].append(epoch_time)
        save_history(history, cfg.experiment_name)
        plot_training_history(history, cfg.experiment_name)

        if val_metrics["loss"] < best_val_loss - cfg.early_stop_min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": vars(cfg),
                    "class_names": class_names,
                    "split_class_names": split_class_names,
                },
                best_model_path,
            )
            print(f"Saved best model to: {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    total_time = time.perf_counter() - training_start
    print(f"Total training time: {format_time(total_time)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
