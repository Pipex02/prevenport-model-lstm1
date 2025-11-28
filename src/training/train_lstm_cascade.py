from __future__ import annotations

"""
Training script for the Component X LSTM cascade (detector + diagnoser).
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required to run train_lstm_cascade.") from exc

from src.data.dataloaders import create_sequence_dataloader
from src.data.datasets import ComponentXSequenceDataset
from src.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig

CASCADE_DEFAULTS = {
    "a": {
        "train_npz": "data/processed/cascade_train_binary.npz",
        "val_npz": "data/processed/cascade_val_binary.npz",
        "output_dir": "artifacts/cascade/detector",
        "num_classes": 2,
    },
    "b": {
        "train_npz": "data/processed/cascade_train_diag.npz",
        "val_npz": "data/processed/cascade_val_diag.npz",
        "output_dir": "artifacts/cascade/diagnoser",
        "num_classes": 4,
    },
}


def normalize_stage(stage: str) -> str:
    stage = stage.lower()
    if stage in {"a", "detector"}:
        return "a"
    if stage in {"b", "diagnoser"}:
        return "b"
    raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM cascade stages.")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["a", "b", "detector", "diagnoser"],
        help="Stage to train: 'a' detector or 'b' diagnoser.",
    )
    parser.add_argument("--train-npz", type=str, default=None, help="Training NPZ.")
    parser.add_argument("--val-npz", type=str, default=None, help="Validation NPZ.")
    parser.add_argument(
        "--feature-stats",
        type=str,
        default="artifacts/feature_stats.json",
        help="Feature stats JSON for normalization.",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-lr-gamma", type=float, default=0.5)
    parser.add_argument("--step-lr-step-size", type=int, default=10)
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="component-x-lstm-cascade",
    )
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    return parser.parse_args()


def apply_stage_defaults(args: argparse.Namespace) -> Tuple[str, Dict[str, str]]:
    stage_key = normalize_stage(args.stage)
    defaults = CASCADE_DEFAULTS[stage_key]
    if args.train_npz is None:
        args.train_npz = defaults["train_npz"]
    if args.val_npz is None:
        args.val_npz = defaults["val_npz"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    if stage_key == "a" and not (args.weighted_sampler or args.class_weighted):
        print("Stage A: enabling weighted sampler (oversampling) by default.")
        args.weighted_sampler = True
    return stage_key, defaults


def select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets_and_loaders(
    args: argparse.Namespace,
) -> Tuple[ComponentXSequenceDataset, Any, Optional[Any]]:
    train_dataset = ComponentXSequenceDataset(
        npz_path=args.train_npz,
        feature_stats_path=args.feature_stats,
        normalize=True,
        device=None,
    )

    train_loader = create_sequence_dataloader(
        npz_path=args.train_npz,
        feature_stats_path=args.feature_stats,
        batch_size=args.batch_size,
        shuffle=not args.weighted_sampler,
        class_weighted=args.weighted_sampler,
        num_workers=0,
        device=None,
    )

    if args.val_npz:
        val_loader = create_sequence_dataloader(
            npz_path=args.val_npz,
            feature_stats_path=args.feature_stats,
            batch_size=args.batch_size,
            shuffle=False,
            class_weighted=False,
            num_workers=0,
            device=None,
        )
    else:
        val_loader = None

    return train_dataset, train_loader, val_loader


def build_model_and_optim(
    args: argparse.Namespace, input_size: int, num_classes: int, device: torch.device
) -> Tuple[LSTMClassifier, torch.optim.Optimizer, Any]:
    config = LSTMClassifierConfig(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        num_classes=num_classes,
        pooling=args.pooling,
    )
    model = LSTMClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
    return model, optimizer, scheduler


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int))
    total = counts.sum()
    num_classes = len(counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c, cnt in enumerate(counts):
        if cnt > 0:
            weights[c] = total / (num_classes * float(cnt))
    if weights.max() == 0:
        weights[:] = 1.0
    else:
        weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> Tuple[int, float, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_metric = float(ckpt.get("best_metric", 0.0))
    return start_epoch, best_metric, ckpt.get("args", {})


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float],
    val_acc: Optional[float],
    val_macro_f1: Optional[float],
    lr: float,
    val_recall: Optional[float] = None,
    val_precision: Optional[float] = None,
) -> None:
    is_new = not metrics_path.exists()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,val_recall,val_precision,lr\n"
            )
        f.write(
            f"{epoch},"
            f"{train_loss:.6f},"
            f"{train_acc:.6f},"
            f"{'' if val_loss is None else f'{val_loss:.6f}'},"  # noqa: E231
            f"{'' if val_acc is None else f'{val_acc:.6f}'},"  # noqa: E231
            f"{'' if val_macro_f1 is None else f'{val_macro_f1:.6f}'},"  # noqa: E231
            f"{'' if val_recall is None else f'{val_recall:.6f}'},"  # noqa: E231
            f"{'' if val_precision is None else f'{val_precision:.6f}'},"  # noqa: E231
            f"{lr:.8f}\n"
        )


def start_mlflow_run(args: argparse.Namespace):
    if not args.mlflow:
        return None
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mlflow is required when --mlflow is enabled.") from exc
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run = mlflow.start_run(run_name=args.mlflow_run_name)
    params = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    mlflow.log_params(params)
    return run


def log_mlflow_metrics(metrics: Dict[str, Optional[float]], step: int) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    if mlflow.active_run() is None:
        return
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, float(value), step=step)


def log_mlflow_artifact(path: Path) -> None:
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        return
    if mlflow.active_run() is None:
        return
    if path.exists():
        mlflow.log_artifact(str(path))


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
    grad_clip: Optional[float],
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (seq, labels, seq_lengths) in enumerate(loader):
        seq = seq.to(device)
        labels = labels.to(device)
        seq_lengths = seq_lengths.to(device)

        optimizer.zero_grad()
        logits = model(seq, seq_lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * seq.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / ((batch_idx + 1) * seq.size(0))
            print(f"  [Batch {batch_idx+1}] avg_loss={avg_loss:.4f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = running_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float, Optional[float], Optional[float]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, labels, seq_lengths in loader:
            seq = seq.to(device)
            labels = labels.to(device)
            seq_lengths = seq_lengths.to(device)

            logits = model(seq, seq_lengths)
            loss = criterion(logits, labels)
            running_loss += loss.item() * seq.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = running_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    val_precision = None
    val_recall = None
    if num_classes == 2:
        val_precision = precision_score(all_targets, all_preds, zero_division=0)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
    return avg_loss, acc, macro_f1, val_precision, val_recall


def main() -> None:
    args = parse_args()
    stage_key, defaults = apply_stage_defaults(args)
    device = select_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader, val_loader = build_datasets_and_loaders(args)

    num_classes = int(train_dataset.labels.max()) + 1
    if train_dataset.labels.min() != 0:
        raise ValueError("Cascade datasets must be zero-indexed (min label == 0).")

    model, optimizer, scheduler = build_model_and_optim(
        args, train_dataset.num_features, num_classes, device
    )

    if args.focal_loss:
        print("Using focal loss (gamma=2.0).")
        criterion = FocalLoss(gamma=2.0).to(device)
    elif args.weighted_loss or args.class_weighted:
        print("Using class-weighted cross-entropy.")
        class_weights = compute_class_weights(train_dataset.labels).to(device)
        print(f"  Weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_metric = 0.0
    metric_name = "recall" if stage_key == "a" else "macro_f1"

    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_metric, _ = load_checkpoint(
                ckpt_path, model, optimizer, scheduler
            )
            print(f"  Resumed at epoch {start_epoch}, best_{metric_name}={best_metric:.4f}")
        else:
            print(f"Checkpoint {ckpt_path} not found; starting from scratch.")

    mlflow_run = start_mlflow_run(args)

    metrics_path = output_dir / "metrics.csv"
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.log_interval,
            grad_clip=args.grad_clip,
        )
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")

        val_loss = None
        val_acc = None
        val_macro_f1 = None
        val_precision = None
        val_recall = None
        if val_loader is not None:
            val_loss, val_acc, val_macro_f1, val_precision, val_recall = evaluate(
                model, val_loader, criterion, device, num_classes
            )
            msg = f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, macro_f1={val_macro_f1:.4f}"
            if val_recall is not None:
                msg += f", recall={val_recall:.4f}"
            print(msg)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        append_metrics_row(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1,
            val_recall=val_recall,
            val_precision=val_precision,
            lr=current_lr,
        )

        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_metric, args)

        log_mlflow_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "lr": current_lr,
            },
            step=epoch,
        )

        candidate_metric = val_recall if stage_key == "a" else val_macro_f1
        if val_loader is not None and candidate_metric is not None and candidate_metric > best_metric:
            best_metric = candidate_metric
            print(f"  New best {metric_name}={best_metric:.4f}; saving checkpoint to {best_ckpt}")
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_metric, args)

    if mlflow_run is not None:
        log_mlflow_artifact(metrics_path)
        log_mlflow_artifact(last_ckpt)
        log_mlflow_artifact(best_ckpt)
        log_mlflow_artifact(Path(args.feature_stats))
        import mlflow

        mlflow.end_run()


if __name__ == "__main__":
    main()
