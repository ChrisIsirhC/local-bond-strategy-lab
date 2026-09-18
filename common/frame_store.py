"""Small Parquet/ZSTD-first table interface used by research workspaces.

Callers retain their historical ``.csv`` logical paths.  This module resolves
the sibling ``.parquet`` first, then falls back to CSV during migration, so a
workspace never needs to retain both representations for application reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def parquet_path(path: Path) -> Path:
    return Path(path).with_suffix(".parquet")


def frame_exists(path: Path) -> bool:
    path = Path(path)
    return parquet_path(path).is_file() or path.is_file()


def read_frame(path: Path, *, parse_dates: Iterable[str] | None = None, **csv_kwargs: object) -> pd.DataFrame:
    """Read the single active table representation, preferring Parquet/ZSTD."""
    path = Path(path)
    parquet = parquet_path(path)
    if parquet.is_file():
        frame = pd.read_parquet(parquet, engine="pyarrow")
    elif path.is_file():
        frame = pd.read_csv(path, encoding="utf-8-sig", **csv_kwargs)
    else:
        raise FileNotFoundError(f"表格不存在：{path}")
    for column in parse_dates or ():
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    """Write only a compressed Parquet representation for a logical CSV path."""
    target = parquet_path(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, engine="pyarrow", compression="zstd", index=False)
    return target
