from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - only hit without torch installed
    raise ImportError(
        "torch is required to use src.data.datasets. "
        "Please install PyTorch before importing this module."
    ) from exc

from src.features.transformer import FeatureTransformer


class ComponentXSequenceDataset(Dataset):
    """PyTorch Dataset wrapping pre-built Component X sequences.

    Data source:
        An .npz file produced by src.features.windowing, containing:
          - sequences: (N, L, F) float32
          - labels: (N,) int64
          - seq_lengths: (N,) int64
          - vehicle_ids: (N,) int64 (optional for analysis)
          - reference_time_step: (N,) float32 (optional for analysis)

    Normalization:
        Optionally applies FeatureTransformer (log1p+z-norm for counters,
        z-norm for histograms, padding reset to 0) on-the-fly in __getitem__.
    """

    def __init__(
        self,
        npz_path: str | Path,
        feature_stats_path: Optional[str | Path] = None,
        normalize: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {self.npz_path}")

        data = np.load(self.npz_path)
        self.sequences = data["sequences"]  # (N, L, F)
        self.labels = data["labels"]  # (N,)
        self.seq_lengths = data["seq_lengths"]  # (N,)
        self.vehicle_ids = data.get("vehicle_ids", None)
        self.reference_time_step = data.get("reference_time_step", None)

        self.num_sequences, self.seq_len, self.num_features = self.sequences.shape

        self.device = torch.device(device) if device is not None else None

        if normalize and feature_stats_path is not None:
            self.transformer = FeatureTransformer.from_json(feature_stats_path)
        else:
            self.transformer = None

    def __len__(self) -> int:
        return self.num_sequences

    def _transform_single(
        self, seq_np: np.ndarray, length: int
    ) -> Tuple[np.ndarray, int]:
        if self.transformer is None:
            return seq_np, int(length)

        # Reuse batch transformer on a singleton batch to keep logic in one place.
        seq_batch = seq_np[None, :, :]  # (1, L, F)
        len_batch = np.asarray([length], dtype=np.int64)
        seq_norm = self.transformer.transform_sequences(seq_batch, len_batch, copy=False)
        return seq_norm[0], int(length)

    def __getitem__(self, idx: int):
        seq_np = self.sequences[idx]  # (L, F)
        label = int(self.labels[idx])
        length = int(self.seq_lengths[idx])

        seq_np, length = self._transform_single(seq_np, length)

        seq = torch.from_numpy(seq_np).float()  # (L, F)
        y = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(length, dtype=torch.long)

        if self.device is not None:
            seq = seq.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            seq_len = seq_len.to(self.device, non_blocking=True)

        return seq, y, seq_len


