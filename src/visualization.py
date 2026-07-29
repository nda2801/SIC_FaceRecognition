import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_experiment_dir(experiment_name):
    experiment_dir = os.path.join("outputs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_history(history, experiment_name):
    experiment_dir = get_experiment_dir(experiment_name)
    history_path = os.path.join(experiment_dir, "history.json")

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    return history_path


def plot_training_history(history, experiment_name):
    experiment_dir = get_experiment_dir(experiment_name)
    figure_path = os.path.join(experiment_dir, "training_curves.png")
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title(f"Loss - {experiment_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["val_positive_distance"],
        label="Positive Distance",
    )
    axes[1].plot(
        epochs,
        history["val_negative_distance"],
        label="Negative Distance",
    )
    axes[1].set_title(f"Validation Distances - {experiment_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    train_rate = [value * 100 for value in history["train_triplet_rate"]]
    val_rate = [value * 100 for value in history["val_triplet_rate"]]
    axes[2].plot(epochs, train_rate, label="Train Triplet Rate")
    axes[2].plot(epochs, val_rate, label="Validation Triplet Rate")
    axes[2].set_title(f"Satisfied Triplets - {experiment_name}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Triplets satisfying margin (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path
