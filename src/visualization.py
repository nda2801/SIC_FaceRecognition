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

    use_triplet_metrics = (
        "val_positive_distance" in history
        and "val_negative_distance" in history
        and "train_triplet_rate" in history
        and "val_triplet_rate" in history
    )

    if use_triplet_metrics:
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        figure, ax = plt.subplots(1, 1, figsize=(6, 5))
        axes = [ax]

    # ===========================
    # Loss
    # ===========================
    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )
    axes[0].set_title(f"Loss - {experiment_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if use_triplet_metrics:

        # ===========================
        # Distance
        # ===========================
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
        axes[1].set_title(
            f"Validation Distances - {experiment_name}"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Euclidean Distance")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # ===========================
        # Triplet Rate
        # ===========================
        train_rate = [
            value * 100
            for value in history["train_triplet_rate"]
        ]

        val_rate = [
            value * 100
            for value in history["val_triplet_rate"]
        ]

        axes[2].plot(
            epochs,
            train_rate,
            label="Train Triplet Rate",
        )
        axes[2].plot(
            epochs,
            val_rate,
            label="Validation Triplet Rate",
        )

        axes[2].set_title(
            f"Satisfied Triplets - {experiment_name}"
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(
            "Triplets satisfying margin (%)"
        )
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    figure.tight_layout()
    figure.savefig(
        figure_path,
        dpi=160,
    )
    plt.close(figure)

    return figure_path


def plot_test_results(
    pair_labels,
    pair_distances,
    roc_curve,
    verification_metrics,
    identification_metrics,
    experiment_name,
):
    experiment_dir = get_experiment_dir(experiment_name)
    figure_path = os.path.join(experiment_dir, "test_results.png")

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

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(
        positive_distances,
        bins=50,
        alpha=0.65,
        density=True,
        label="Same identity",
    )

    axes[0].hist(
        negative_distances,
        bins=50,
        alpha=0.65,
        density=True,
        label="Different identities",
    )

    axes[0].axvline(
        verification_metrics["eer_distance_threshold"],
        color="black",
        linestyle="--",
        label="EER threshold",
    )

    axes[0].set_title(
        "Verification Distance Distribution"
    )
    axes[0].set_xlabel("Euclidean Distance")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        roc_curve["false_positive_rate"],
        roc_curve["true_positive_rate"],
        label=f"AUC={verification_metrics['roc_auc']:.4f}",
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Acceptance Rate")
    axes[1].set_ylabel("True Acceptance Rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    metric_names = [
        "Recall@1",
        "Recall@5",
        "mAP",
    ]

    metric_values = [
        identification_metrics["recall_at_1"] * 100,
        identification_metrics["recall_at_5"] * 100,
        identification_metrics["mean_average_precision"] * 100,
    ]

    bars = axes[2].bar(
        metric_names,
        metric_values,
    )

    axes[2].set_title(
        "Gallery-Probe Identification"
    )
    axes[2].set_ylabel("Score (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].bar_label(
        bars,
        fmt="%.2f",
    )

    figure.suptitle(
        f"Test Evaluation - {experiment_name}"
    )

    figure.tight_layout()
    figure.savefig(
        figure_path,
        dpi=160,
    )
    plt.close(figure)

    return figure_path


def save_test_results(results, experiment_name):
    experiment_dir = get_experiment_dir(experiment_name)

    results_path = os.path.join(
        experiment_dir,
        "test_results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    return results_path