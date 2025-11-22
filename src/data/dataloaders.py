from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover - only hit without torch installed
    raise ImportError(
        "torch is required to use src.data.dataloaders. "
        "Please install PyTorch before importing this module."
    ) from exc

from src.data.datasets import ComponentXSequenceDataset


def create_sequence_dataloader(
    npz_path: str | Path,
    feature_stats_path: Optional[str | Path] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    class_weighted: bool = False,
    num_workers: int = 0,
    device: Optional[torch.device | str] = None,
) -> DataLoader:
    """Create a DataLoader for Component X sequence data.

    Args:
        npz_path: Path to .npz file with sequences/labels/seq_lengths.
        feature_stats_path: Path to feature_stats.json (for normalization).
        batch_size: Batch size.
        shuffle: Whether to shuffle (ignored if class_weighted=True).
        class_weighted: If True, use WeightedRandomSampler based on class frequencies.
        num_workers: DataLoader num_workers.
        device: Optional torch.device or string ("cuda", "cpu"); passed to Dataset.

    Returns:
        A configured DataLoader instance.
    """
    dataset = ComponentXSequenceDataset(
        npz_path=npz_path,
        feature_stats_path=feature_stats_path,
        normalize=feature_stats_path is not None,
        device=device,
    )

    if class_weighted:
        labels_np = dataset.labels.astype(int)
        class_counts = np.bincount(labels_np)
        # Avoid division by zero; unseen classes get weight 0.
        class_weights = np.zeros_like(class_counts, dtype=np.float32)
        for cls, cnt in enumerate(class_counts):
            if cnt > 0:
                class_weights[cls] = 1.0 / float(cnt)

        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device is not None and "cuda" in str(device),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device is not None and "cuda" in str(device),
        )

    return loader

