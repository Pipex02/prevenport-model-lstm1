from __future__ import annotations

"""
Evaluation utilities for Phase 8 of the Component X LSTM project.

This script loads a trained checkpoint and produces the artifacts requested in
the project plan:
    - Classification report with per-class precision/recall/F1/support.
    - Confusion matrix (numbers + plot).
    - Per-class ROC and Precision-Recall curves.
    - Reliability / calibration diagram for top-1 probabilities.

All metrics and plots are written to the specified output directory so the
notebooks/docs can embed them without recomputing results by hand.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib optional until plotting
    plt = None
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - tool/runtime without torch
    raise ImportError(
        "torch is required to run the evaluation script. "
        "Install PyTorch before executing src/training/evaluate.py."
    ) from exc

from src.data.datasets import ComponentXSequenceDataset
from src.data.labels import PROXIMITY_CLASS_NAMES
from src.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Component X LSTM checkpoint on a dataset."
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to sequences .npz file (train/val/test).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint produced by train_lstm.py.",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default="artifacts/feature_stats.json",
        help="Feature stats JSON for normalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluations",
        help="Directory where metrics/plots will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu/cuda). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        help="Number of bins for reliability diagram.",
    )
    return parser.parse_args()


def _bool_from_checkpoint(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_model(
    checkpoint_path: Path,
    input_size: int,
    device: torch.device,
) -> LSTMClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args: Dict[str, object] = checkpoint.get("args", {})

    config = LSTMClassifierConfig(
        input_size=input_size,
        hidden_size=int(ckpt_args.get("hidden_size", 128)),
        num_layers=int(ckpt_args.get("num_layers", 2)),
        bidirectional=_bool_from_checkpoint(ckpt_args.get("bidirectional", True)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        num_classes=5,
        pooling=str(ckpt_args.get("pooling", "last")),
    )

    model = LSTMClassifier(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def get_device(preferred: Optional[str]) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(
    npz_path: Path,
    feature_stats: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ComponentXSequenceDataset(
        npz_path=npz_path,
        feature_stats_path=feature_stats,
        normalize=True,
        device=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader


def run_inference(
    model: LSTMClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    with torch.no_grad():
        for sequences, labels, seq_lengths in dataloader:
            sequences = sequences.to(device)
            seq_lengths = seq_lengths.to(device)

            logits = model(sequences, seq_lengths)
            probs = torch.softmax(logits, dim=1)

            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())

    y_prob = np.concatenate(probs_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)
    y_pred = y_prob.argmax(axis=1)
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def compute_reliability_curve(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bins: int,
) -> List[Dict[str, float]]:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(confidences, bin_edges, right=True)
    reliability: List[Dict[str, float]] = []
    for b in range(1, len(bin_edges)):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correctness[mask].mean())
        reliability.append(
            {
                "bin_start": float(bin_edges[b - 1]),
                "bin_end": float(bin_edges[b]),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "count": int(mask.sum()),
            }
        )
    return reliability


def save_classification_report(
    report: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    if plt is None:
        raise ImportError("matplotlib is required to plot the confusion matrix.")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_curves(
    curve_dict: Dict[str, Dict[str, List[float]]],
    kind: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required to plot ROC/PR curves.")
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_name, data in curve_dict.items():
        label = f"{class_name} (AUC={data['auc']:.3f})"
        ax.plot(data["x"], data["y"], label=label)
    if kind == "roc":
        ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{kind.upper()} Curves")
    ax.legend(loc="lower right" if kind == "roc" else "upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_reliability_curve(
    reliability: List[Dict[str, float]], output_path: Path
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required to plot the reliability diagram.")
    if not reliability:
        return
    confidences = [row["avg_confidence"] for row in reliability]
    accuracies = [row["avg_accuracy"] for row in reliability]
    bin_centers = [
        (row["bin_start"] + row["bin_end"]) / 2.0 for row in reliability
    ]
    widths = [row["bin_end"] - row["bin_start"] for row in reliability]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_centers, accuracies, width=widths, alpha=0.7, label="Observed")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram (Top-1 Probabilities)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    npz_path = Path(args.npz)
    feature_stats = Path(args.feature_stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(
        npz_path=npz_path,
        feature_stats=feature_stats,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dataset = dataloader.dataset  # type: ignore[attr-defined]
    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        input_size=dataset.num_features,  # type: ignore[attr-defined]
        device=device,
    )

    outputs = run_inference(model, dataloader, device)
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    y_prob = outputs["y_prob"]

    labels = sorted(PROXIMITY_CLASS_NAMES.keys())
    class_names_verbose = [PROXIMITY_CLASS_NAMES[i] for i in labels]
    class_labels_numeric = [str(i) for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names_verbose,
        zero_division=0,
        output_dict=True,
    )
    save_classification_report(report, output_dir / "classification_report.json")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_matrix_plot(cm, class_labels_numeric, output_dir / "confusion_matrix.png")

    overall_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    y_true_onehot = label_binarize(y_true, classes=labels)
    roc_curves: Dict[str, Dict[str, List[float]]] = {}
    pr_curves: Dict[str, Dict[str, List[float]]] = {}

    for idx, class_id in enumerate(labels):
        positives = y_true_onehot[:, idx].sum()
        negatives = y_true_onehot.shape[0] - positives
        if positives == 0 or negatives == 0:
            continue  # Cannot compute curve for degenerate class

        fpr, tpr, _ = roc_curve(y_true_onehot[:, idx], y_prob[:, idx])
        prec, rec, _ = precision_recall_curve(y_true_onehot[:, idx], y_prob[:, idx])
        class_key = str(class_id)
        roc_curves[class_key] = {
            "x": fpr.tolist(),
            "y": tpr.tolist(),
            "auc": auc(fpr, tpr),
        }
        pr_curves[class_key] = {
            "x": rec.tolist(),
            "y": prec.tolist(),
            "auc": auc(rec, prec),
        }

    with (output_dir / "roc_curves.json").open("w", encoding="utf-8") as f:
        json.dump(roc_curves, f, indent=2)
    with (output_dir / "pr_curves.json").open("w", encoding="utf-8") as f:
        json.dump(pr_curves, f, indent=2)

    if roc_curves:
        plot_curves(
            roc_curves,
            kind="roc",
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            output_path=output_dir / "roc_curves.png",
        )
    if pr_curves:
        plot_curves(
            pr_curves,
            kind="pr",
            x_label="Recall",
            y_label="Precision",
            output_path=output_dir / "pr_curves.png",
        )

    max_conf = y_prob.max(axis=1)
    correctness = (y_pred == y_true).astype(float)
    reliability = compute_reliability_curve(
        confidences=max_conf,
        correctness=correctness,
        bins=args.calibration_bins,
    )
    with (output_dir / "reliability.json").open("w", encoding="utf-8") as f:
        json.dump(reliability, f, indent=2)
    plot_reliability_curve(reliability, output_dir / "reliability.png")

    print(
        f"Evaluation complete. Outputs written to {output_dir.resolve()}",
    )


if __name__ == "__main__":
    main()
