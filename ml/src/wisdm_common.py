"""Shared loading, preprocessing, and evaluation helpers for WISDM."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

WINDOW_SIZE: Final = 26
AXES: Final = ["x", "y", "z"]
LABELS: Final = ["stationary", "walking", "running"]
LABEL_MAP: Final = {
    "Sitting": "stationary",
    "Standing": "stationary",
    "Walking": "walking",
    "Jogging": "running",
}


def load_wisdm(path: str | Path) -> pd.DataFrame:
    """Load WISDM raw data and retain the three activities used by the model."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"WISDM file not found: {source}")

    frame = pd.read_csv(
        source,
        header=None,
        names=["user", "activity", "timestamp", "x", "y", "z"],
        sep=",",
        engine="python",
        on_bad_lines="skip",
    )
    frame["z"] = frame["z"].astype(str).str.rstrip(";")
    frame["activity"] = frame["activity"].map(LABEL_MAP)
    frame = frame.dropna(subset=["activity", *AXES]).copy()
    frame[AXES] = frame[AXES].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=AXES)
    frame["user"] = pd.to_numeric(frame["user"], errors="raise").astype(int)
    frame = frame.sort_values(["user", "timestamp"]).reset_index(drop=True)
    return frame


def make_windows(
    frame: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping windows that never cross users or activity changes."""
    values: list[np.ndarray] = []
    labels: list[str] = []
    users: list[int] = []
    for (user, activity), group in frame.groupby(["user", "activity"], sort=False):
        samples = group[AXES].to_numpy(dtype=np.float32)
        count = len(samples) // window_size
        for index in range(count):
            start = index * window_size
            values.append(samples[start : start + window_size])
            labels.append(activity)
            users.append(int(user))
    if not values:
        raise ValueError("No complete windows could be created from the dataset.")
    return np.stack(values), np.asarray(labels), np.asarray(users)


def normalize_windows(x: np.ndarray) -> np.ndarray:
    """Apply the firmware-compatible full-scale normalization."""
    return x / 4000.0


def split_by_user(
    x: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split whole users, rather than individual windows, into train and test sets."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(x, y, groups=users))
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def encode_labels(labels: np.ndarray) -> np.ndarray:
    indices = {label: index for index, label in enumerate(LABELS)}
    return np.asarray([indices[label] for label in labels], dtype=np.int64)
