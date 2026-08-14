import os
import sys
import random
import time
from pathlib import Path

# Automatic Path Bootstrap
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0] if CURRENT_DIR.name == "src" else CURRENT_DIR
for p in [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from PIL import Image

from config import get_parser
from data import build_dataloaders
from model import build_model
from arcface import ArcFaceLoss
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
    print(f"[Train ArcFace] Using device: {device}")
    return device


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    criterion.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
    }


def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    cfg = get_parser()
    # Continuous ArcFace experiment name
    if cfg.experiment_name == "sic_facevit_infonce_v2":
        cfg.experiment_name = "sic_facevit_arcface_v1"

    set_seed(cfg.seed)
    device = get_device()
    loaders, class_names, split_class_names = build_dataloaders(cfg)

    num_classes = len(class_names)
    print(f"[ArcFace Setup] Total training identities (num_classes): {num_classes}")

    # Build Vision Transformer model
    model = build_model(cfg).to(device)

    # Initialize ArcFace Loss Module (Angular Margin m = 0.35 rad cho do tong quat cao, Scale s = 30.0)
    in_feat = getattr(cfg, "face_embedding_dim", getattr(cfg, "embed_dim", 128))
    margin = getattr(cfg, "arcface_margin", 0.35)
    criterion = ArcFaceLoss(
        in_features=in_feat,
        num_classes=num_classes,
        s=30.0,
        m=margin,
    ).to(device)

    # Optimizer trains both ViT model weights and ArcFace class weights with weight decay
    weight_decay = getattr(cfg, "weight_decay", 1e-3)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg.lr},
            {"params": criterion.parameters(), "lr": cfg.lr},
        ],
        weight_decay=weight_decay,
    )

    # StepLR: Giam Learning Rate 50% sau moi 8 epoch de ep Val Loss sut giam va chong Overfitting
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("src/checkpoints", exist_ok=True)

    best_model_path = os.path.join("checkpoints", f"{cfg.experiment_name}_best.pth")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
        "lr": [],
    }

    print(f"=== STARTING ARCFACE TRAINING EXPERIMENT: {cfg.experiment_name} (Margin m={margin}, Cosine LR Scheduler) ===")
    train_loader = loaders.get("train_triplet", loaders.get("train"))
    val_loader = loaders.get("val_triplet", loaders.get("val"))

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Val Loss={val_metrics['loss']:.4f} "
            f"LR={current_lr:.6f} "
            f"Time={format_time(epoch_time)}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["epoch_time"].append(epoch_time)

        save_history(history, cfg.experiment_name)
        plot_training_history(history, cfg.experiment_name)

        if val_metrics["loss"] < best_val_loss - cfg.early_stop_min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(cfg),
                "class_names": class_names,
                "split_class_names": split_class_names,
            }

            torch.save(checkpoint_data, best_model_path)
            torch.save(checkpoint_data, f"src/checkpoints/{cfg.experiment_name}_best.pth")
            print(f"✅ Saved best ArcFace model to: {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    total_time = time.perf_counter() - training_start
    print(f"\n=== ARCFACE TRAINING FINISHED ===")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Model Checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()