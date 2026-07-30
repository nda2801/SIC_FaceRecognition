import os
import random
import time

import torch

from config import get_parser
from data import build_dataloaders
from model import build_model
from visualization import plot_training_history, save_history
from infonce import InfoNCELoss

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

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(
            embeddings,
            labels,
        )
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = model(images)
        loss = criterion(
            embeddings,
            labels,
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
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
    criterion = InfoNCELoss(
        temperature=cfg.temperature,
    ).to(device)
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
        )
        val_metrics = evaluate(
            model,
            loaders["val_triplet"],
            criterion,
            device,
        )
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] "
            f"Train Loss={train_metrics['loss']:.4f} "
            f"Val Loss={val_metrics['loss']:.4f} "
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