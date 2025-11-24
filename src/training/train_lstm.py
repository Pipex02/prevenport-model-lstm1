from __future__ import annotations

"""
Training script for the Component X LSTM classifier (Phase 7).

This module wires together:
- ComponentXSequenceDataset + DataLoaders (Phase 5)
- FeatureTransformer-based normalization (Phase 4)
- LSTMClassifier model (Phase 6)

It supports:
- Cross-entropy loss with optional class weights.
- Basic train/validation loop with accuracy and macro F1.
- Checkpointing:
    - best.pt  : best validation macro F1
    - last.pt  : last epoch state (for resume)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR
except ImportError as exc:  # pragma: no cover - only hit without torch installed
    raise ImportError(
        "torch is required to run train_lstm. "
        "Please install PyTorch before using this script."
    ) from exc

from src.data.dataloaders import create_sequence_dataloader
from src.data.datasets import ComponentXSequenceDataset
from src.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM classifier for Component X.")

    # Data
    parser.add_argument(
        "--train-npz",
        type=str,
        default="data/train_sequences.npz",
        help="Path to training sequences .npz file.",
    )
    parser.add_argument(
        "--val-npz",
        type=str,
        default=None,
        help="Path to validation sequences .npz file (optional but recommended).",
    )
    parser.add_argument(
        "--feature-stats",
        type=str,
        default="artifacts/feature_stats.json",
        help="Path to feature_stats.json for normalization.",
    )

    # Model
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="LSTM hidden size.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional LSTM.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for LSTM and classifier head.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling strategy over time dimension.",
    )

    # Optimization
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--step-lr-gamma",
        type=float,
        default=0.5,
        help="Gamma for StepLR scheduler.",
    )
    parser.add_argument(
        "--step-lr-step-size",
        type=int,
        default=10,
        help="Step size for StepLR scheduler (in epochs).",
    )
    parser.add_argument(
        "--class-weighted",
        action="store_true",
        help="Use class-weighted sampling and/or loss.",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu'). If None, auto-select.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="How many batches to wait before logging training status.",
    )

    return parser.parse_args()


def select_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets_and_loaders(
    args: argparse.Namespace, device: torch.device
) -> Tuple[ComponentXSequenceDataset, Any, Optional[Any]]:
    # Build train dataset directly for class weight calculation.
    train_dataset = ComponentXSequenceDataset(
        npz_path=args.train_npz,
        feature_stats_path=args.feature_stats,
        normalize=True,
        device=None,  # DataLoader will handle device transfer
    )

    # Train DataLoader (with optional class weighting sampler).
    train_loader = create_sequence_dataloader(
        npz_path=args.train_npz,
        feature_stats_path=args.feature_stats,
        batch_size=args.batch_size,
        shuffle=not args.class_weighted,
        class_weighted=args.class_weighted,
        num_workers=0,
        device=None,
    )

    # Validation DataLoader (if provided).
    if args.val_npz is not None:
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
    args: argparse.Namespace, input_size: int, device: torch.device
) -> Tuple[LSTMClassifier, torch.optim.Optimizer, Any]:
    config = LSTMClassifierConfig(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        num_classes=5,
        pooling=args.pooling,
    )
    model = LSTMClassifier(config).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(
        optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma
    )

    return model, optimizer, scheduler


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int))
    num_classes = len(counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c, cnt in enumerate(counts):
        if cnt > 0:
            weights[c] = 1.0 / float(cnt)
    if weights.max() == 0:
        # Fallback to uniform weights if something went wrong
        weights[:] = 1.0
    return torch.from_numpy(weights)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_f1: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(state, path)


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float],
    val_acc: Optional[float],
    val_macro_f1: Optional[float],
    lr: float,
) -> None:
    is_new = not metrics_path.exists()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr\n"
            )
        f.write(
            f"{epoch},"
            f"{train_loss:.6f},"
            f"{train_acc:.6f},"
            f"{'' if val_loss is None else f'{val_loss:.6f}'},"
            f"{'' if val_acc is None else f'{val_acc:.6f}'},"
            f"{'' if val_macro_f1 is None else f'{val_macro_f1:.6f}'},"
            f"{lr:.8f}\n"
        )


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
    best_val_f1 = float(ckpt.get("best_val_f1", 0.0))
    return start_epoch, best_val_f1, ckpt.get("args", {})


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
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
) -> Tuple[float, float, float]:
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
    return avg_loss, acc, macro_f1


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_dataset, train_loader, val_loader = build_datasets_and_loaders(args, device)

    # Model & optimizer
    input_size = train_dataset.num_features
    model, optimizer, scheduler = build_model_and_optim(args, input_size, device)

    # Loss
    if args.class_weighted:
        class_weights = compute_class_weights(train_dataset.labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_f1 = 0.0

    # Optional resume
    if args.resume_from is not None:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            print(f"Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_val_f1, ckpt_args = load_checkpoint(
                ckpt_path, model, optimizer, scheduler
            )
            print(f"  Resumed at epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")
        else:
            print(f"Checkpoint {ckpt_path} not found; starting from scratch.")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.log_interval
        )
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")

        val_loss = None
        val_acc = None
        if val_loader is not None:
            val_loss, val_acc, val_macro_f1 = evaluate(
                model, val_loader, criterion, device
            )
            print(
                f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}, "
                f"macro_f1={val_macro_f1:.4f}"
            )
        else:
            val_macro_f1 = 0.0

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Log metrics for this epoch
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_path = output_dir / "metrics.csv"
        append_metrics_row(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_macro_f1 if val_loader is not None else None,
            lr=current_lr,
        )

        # Save last checkpoint
        last_ckpt = output_dir / "last.pt"
        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_val_f1, args)

        # Save best checkpoint based on validation macro F1
        if val_loader is not None and val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_ckpt = output_dir / "best.pt"
            print(
                f"  New best macro F1={best_val_f1:.4f}; saving checkpoint to {best_ckpt}"
            )
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_val_f1, args)


if __name__ == "__main__":
    main()
