"""Prepare WISDM windows for CNN and Random Forest experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wisdm_common import load_wisdm, make_windows, normalize_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to WISDM_ar_v1.1_raw.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/data/wisdm_windows.npz"),
    )
    args = parser.parse_args()

    frame = load_wisdm(args.dataset)
    x, labels, users = make_windows(frame)
    x = normalize_windows(x)
    y = np.asarray(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, users=users)
    print(f"Saved {len(y)} windows to {args.output}")
    print(frame["activity"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
