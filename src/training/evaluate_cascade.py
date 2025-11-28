from __future__ import annotations

"""
Evaluation script for the Component X cascade (detector + diagnoser).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required to run evaluate_cascade.") from exc

from src.data.datasets import ComponentXSequenceDataset
from src.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig

STAGE_DEFAULTS = {
    "detector": {
        "val_npz": "data/processed/cascade_val_binary.npz",
        "checkpoint": "artifacts/cascade/detector/best.pt",
        "output_dir": "artifacts/evaluations/cascade/detector",
        "num_classes": 2,
    },
    "diagnoser": {
        "val_npz": "data/processed/cascade_val_diag.npz",
        "checkpoint": "artifacts/cascade/diagnoser/best.pt",
        "output_dir": "artifacts/evaluations/cascade/diagnoser",
        "num_classes": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cascade models.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["detector", "diagnoser", "cascade", "all"],
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--detector-npz",
        type=str,
        default=None,
        help="Validation NPZ for detector evaluation.",
    )
    parser.add_argument(
        "--diagnoser-npz",
        type=str,
        default=None,
        help="Validation NPZ for diagnoser evaluation.",
    )
    parser.add_argument(
        "--cascade-npz",
        type=str,
        default="data/processed/val_sequences.npz",
        help="Full validation NPZ (5-class) for cascade evaluation.",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=str,
        default=None,
        help="Detector checkpoint path.",
    )
    parser.add_argument(
        "--diagnoser-checkpoint",
        type=str,
        default=None,
        help="Diagnoser checkpoint path.",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default="artifacts/feature_stats.json",
        help="Feature statistics JSON for normalization.",
    )
    parser.add_argument(
        "--detector-output-dir",
        type=str,
        default=None,
        help="Where to store detector metrics.",
    )
    parser.add_argument(
        "--diagnoser-output-dir",
        type=str,
        default=None,
        help="Where to store diagnoser metrics.",
    )
    parser.add_argument(
        "--cascade-output-dir",
        type=str,
        default="artifacts/evaluations/cascade/pipeline",
        help="Where to store cascade metrics.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detector probability threshold for cascade evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation loaders.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def apply_stage_defaults(args: argparse.Namespace) -> None:
    det_defaults = STAGE_DEFAULTS["detector"]
    diag_defaults = STAGE_DEFAULTS["diagnoser"]
    if args.detector_npz is None:
        args.detector_npz = det_defaults["val_npz"]
    if args.detector_checkpoint is None:
        args.detector_checkpoint = det_defaults["checkpoint"]
    if args.detector_output_dir is None:
        args.detector_output_dir = det_defaults["output_dir"]
    if args.diagnoser_npz is None:
        args.diagnoser_npz = diag_defaults["val_npz"]
    if args.diagnoser_checkpoint is None:
        args.diagnoser_checkpoint = diag_defaults["checkpoint"]
    if args.diagnoser_output_dir is None:
        args.diagnoser_output_dir = diag_defaults["output_dir"]


def select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: Path,
    input_size: int,
    num_classes: int,
    device: torch.device,
) -> LSTMClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ck_args = ckpt.get("args", {})
    config = LSTMClassifierConfig(
        input_size=input_size,
        hidden_size=int(ck_args.get("hidden_size", 128)),
        num_layers=int(ck_args.get("num_layers", 2)),
        bidirectional=bool(ck_args.get("bidirectional", True)),
        dropout=float(ck_args.get("dropout", 0.1)),
        num_classes=num_classes,
        pooling=ck_args.get("pooling", "last"),
    )
    model = LSTMClassifier(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, cm, fmt="%d", delimiter=",")


def evaluate_stage(
    dataset_path: str,
    checkpoint_path: str,
    feature_stats: str,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
) -> None:
    dataset = ComponentXSequenceDataset(
        npz_path=dataset_path,
        feature_stats_path=feature_stats,
        normalize=True,
        device=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_model(Path(checkpoint_path), dataset.num_features, num_classes, device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, labels, seq_lengths in loader:
            seq = seq.to(device)
            seq_lengths = seq_lengths.to(device)
            logits = model(seq, seq_lengths)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    if num_classes == 2:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    save_json(metrics, output_dir / "summary_metrics.json")
    save_json(report, output_dir / "classification_report.json")
    save_confusion_matrix(cm, [str(i) for i in range(num_classes)], output_dir / "confusion_matrix.csv")
    print(f"Stage results written to {output_dir}")


def evaluate_cascade_pipeline(
    detector_ckpt: str,
    diagnoser_ckpt: str,
    cascade_npz: str,
    feature_stats: str,
    threshold: float,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    dataset = ComponentXSequenceDataset(
        npz_path=cascade_npz,
        feature_stats_path=feature_stats,
        normalize=True,
        device=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    detector = load_model(Path(detector_ckpt), dataset.num_features, 2, device)
    diagnoser = load_model(Path(diagnoser_ckpt), dataset.num_features, 4, device)

    softmax = torch.nn.Softmax(dim=1)
    y_true_full = []
    y_pred_full = []
    y_true_detector = []
    y_pred_detector = []
    positive_indices: list[int] = []

    with torch.no_grad():
        offset = 0
        for seq, labels, seq_lengths in loader:
            seq = seq.to(device)
            seq_lengths = seq_lengths.to(device)
            logits = detector(seq, seq_lengths)
            probs = softmax(logits)
            detector_preds = (probs[:, 1] >= threshold).cpu().numpy().astype(int)
            y_pred_detector.append(detector_preds)
            binary_labels = (labels.numpy() > 0).astype(int)
            y_true_detector.append(binary_labels)

            batch_size_actual = labels.size(0)
            for i in range(batch_size_actual):
                global_idx = offset + i
                if detector_preds[i] == 1:
                    positive_indices.append(global_idx)
            offset += batch_size_actual

    y_true_detector = np.concatenate(y_true_detector)
    y_pred_detector = np.concatenate(y_pred_detector)

    # Prepare diagnoser predictions
    y_pred_diag = np.zeros(len(dataset), dtype=int)
    if positive_indices:
        diagnoser.eval()
        diag_batch = []
        diag_lengths = []
        diag_indices = []
        with torch.no_grad():
            for idx in positive_indices:
                seq, _, seq_len = dataset[idx]
                diag_batch.append(seq.unsqueeze(0))
                diag_lengths.append(seq_len.unsqueeze(0))
                diag_indices.append(idx)
                if len(diag_batch) == batch_size:
                    batch_tensor = torch.cat(diag_batch, dim=0).to(device)
                    length_tensor = torch.cat(diag_lengths, dim=0).to(device)
                    logits = diagnoser(batch_tensor, length_tensor)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    for out_idx, dataset_idx in enumerate(diag_indices):
                        y_pred_diag[dataset_idx] = int(preds[out_idx] + 1)
                    diag_batch, diag_lengths, diag_indices = [], [], []
            if diag_batch:
                batch_tensor = torch.cat(diag_batch, dim=0).to(device)
                length_tensor = torch.cat(diag_lengths, dim=0).to(device)
                logits = diagnoser(batch_tensor, length_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for out_idx, dataset_idx in enumerate(diag_indices):
                    y_pred_diag[dataset_idx] = int(preds[out_idx] + 1)

    y_true_full = dataset.labels
    y_pred_full = np.where(y_pred_diag > 0, y_pred_diag, 0)

    metrics = {
        "detector_accuracy": accuracy_score(y_true_detector, y_pred_detector),
        "detector_precision": precision_score(y_true_detector, y_pred_detector, zero_division=0),
        "detector_recall": recall_score(y_true_detector, y_pred_detector, zero_division=0),
        "cascade_accuracy": accuracy_score(y_true_full, y_pred_full),
        "cascade_macro_f1": f1_score(y_true_full, y_pred_full, average="macro"),
    }
    combined_report = classification_report(
        y_true_full,
        y_pred_full,
        labels=[0, 1, 2, 3, 4],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_full, y_pred_full, labels=[0, 1, 2, 3, 4])

    save_json(metrics, output_dir / "summary_metrics.json")
    save_json(combined_report, output_dir / "classification_report.json")
    save_confusion_matrix(cm, [str(i) for i in range(5)], output_dir / "confusion_matrix.csv")
    print(f"Cascade evaluation written to {output_dir}")


def main() -> None:
    args = parse_args()
    apply_stage_defaults(args)
    device = select_device(args.device)
    stages = {args.stage} if args.stage != "all" else {"detector", "diagnoser", "cascade"}

    if "detector" in stages:
        evaluate_stage(
            dataset_path=args.detector_npz,
            checkpoint_path=args.detector_checkpoint,
            feature_stats=args.feature_stats,
            batch_size=args.batch_size,
            device=device,
            output_dir=Path(args.detector_output_dir),
            num_classes=2,
        )

    if "diagnoser" in stages:
        evaluate_stage(
            dataset_path=args.diagnoser_npz,
            checkpoint_path=args.diagnoser_checkpoint,
            feature_stats=args.feature_stats,
            batch_size=args.batch_size,
            device=device,
            output_dir=Path(args.diagnoser_output_dir),
            num_classes=4,
        )

    if "cascade" in stages:
        evaluate_cascade_pipeline(
            detector_ckpt=args.detector_checkpoint,
            diagnoser_ckpt=args.diagnoser_checkpoint,
            cascade_npz=args.cascade_npz,
            feature_stats=args.feature_stats,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            output_dir=Path(args.cascade_output_dir),
        )


if __name__ == "__main__":
    main()
