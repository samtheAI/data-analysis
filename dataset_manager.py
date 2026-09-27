from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ROWS = 500_000
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


@dataclass
class Dataset:
    dataset_id: str
    name: str
    frame: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{self.name} · {len(self.frame):,} rows × {len(self.frame.columns):,} columns"


def _safe_stem(filename: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", Path(filename).stem).strip("_").lower()
    return stem[:45] or "dataset"


def load_dataset(filename: str, file_obj: BinaryIO) -> Dataset:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")

    raw = file_obj.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("The file is larger than the 25 MB upload limit.")

    buffer = io.BytesIO(raw)
    try:
        if extension == ".csv":
            frame = pd.read_csv(buffer)
        elif extension in {".xlsx", ".xls"}:
            frame = pd.read_excel(buffer)
        elif extension == ".json":
            try:
                frame = pd.read_json(buffer)
            except ValueError:
                buffer.seek(0)
                frame = pd.read_json(buffer, lines=True)
        else:
            frame = pd.read_parquet(buffer)
    except Exception as exc:
        raise ValueError(f"Could not read {filename}: {exc}") from exc

    if frame.empty:
        raise ValueError(f"{filename} contains no data rows.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"{filename} exceeds the {MAX_ROWS:,}-row classroom limit.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not supported: {duplicates}")

    frame.columns = [str(column) for column in frame.columns]
    digest = hashlib.sha256(raw).hexdigest()[:10]
    return Dataset(f"{_safe_stem(filename)}_{digest}", filename, frame)


def schema_for_agent(dataset: Dataset) -> dict:
    """Create useful metadata without exposing any raw records to the model."""
    frame = dataset.frame
    columns = []
    for name in frame.columns:
        series = frame[name]
        item = {
            "name": name,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            item["statistics"] = {
                "min": _json_number(numeric.min()),
                "max": _json_number(numeric.max()),
                "mean": _json_number(numeric.mean()),
            }
        columns.append(item)
    return {
        "dataset_id": dataset.dataset_id,
        "original_filename": dataset.name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
    }


def _json_number(value):
    if pd.isna(value):
        return None
    return round(float(value), 6)

