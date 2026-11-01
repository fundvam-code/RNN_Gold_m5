"""Canonical preprocessing (v2) for the 20-bar, 12-feature BUY ONNX contract.

Input : CSV exported by PrepareData.mq5 v2.x (Common\\Files\\data_BUY_*.csv)
        Each row = one BUY signal with a 20-bar M5 window (lag0 = newest).
Columns: datetime, signal, lag0_*...lag19_*, result, label
Raw per-bar features (12): open, high, low, close, direction,
        ema8, ema21, rsi, atr, bb_upper, bb_mid, bb_lower
label  : "[1,0]" -> сделка закрылась по TP => класс 1 (прибыль)
         "[0,1]" -> иначе (SL / удержание / конец дня) => класс 0

Output: artifacts/data_BUY_*_normalized.npz with
        X [N,20,12] float32, y int64 (0/1), result float64, datetime str
        + model_features.json (contract fingerprint)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_BARS = 20
RAW_FEATURES = 12
MODEL_FEATURES = 12
EPS = 1e-6

RAW_NAMES = (
    "open", "high", "low", "close", "direction",
    "ema8", "ema21", "rsi", "atr",
    "bb_upper", "bb_mid", "bb_lower",
)
MODEL_NAMES = RAW_NAMES   # v2: никаких производных признаков — сырые колонки как есть

# Каналы НЕ нормализуются (категориальные/ограниченные):
#  4 = direction (-1/1) остаётся как есть
NO_NORM_CHANNELS = (4,)
NORMALIZE_MASK = tuple(i for i in range(RAW_FEATURES) if i not in NO_NORM_CHANNELS)


def expected_columns() -> list[str]:
    columns = ["datetime", "signal"]
    columns.extend(f"lag{lag}_{name}" for lag in range(RAW_BARS) for name in RAW_NAMES)
    columns.extend(("result", "label"))
    return columns


def schema_fingerprint() -> str:
    return hashlib.sha256("\n".join(expected_columns()).encode("utf-8")).hexdigest()


def parse_label(raw) -> int:
    """'[1,0]' -> 1 (положительный, закрытие по TP), всё остальное -> 0."""
    s = str(raw).strip().lower().replace(" ", "")
    return 1 if s in ("[1,0]", "'[1,0]'") else 0


def validate_frame(frame: pd.DataFrame, source: str = "CSV") -> None:
    required = {"datetime", "signal", "result", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    actual_lags = [c for c in frame.columns if c.startswith("lag")]
    expected_lags = expected_columns()[2:-2]
    if actual_lags != expected_lags:
        if len(actual_lags) != len(expected_lags):
            raise ValueError(f"{source}: expected {len(expected_lags)} lag columns, got {len(actual_lags)}")
        raise ValueError(f"{source}: lag column order/schema differs from canonical contract")
    if len(actual_lags) != RAW_BARS * RAW_FEATURES:
        raise ValueError(f"{source}: expected {RAW_BARS}x{RAW_FEATURES} raw fields")


def load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_frame(frame, str(path))
    frame["datetime"] = pd.to_datetime(frame["datetime"], format="%Y.%m.%d %H:%M", errors="raise")
    frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    return frame


def raw_tensor(frame: pd.DataFrame) -> np.ndarray:
    values = frame[[f"lag{lag}_{name}" for lag in range(RAW_BARS) for name in RAW_NAMES]].to_numpy(dtype=np.float64)
    raw = values.reshape(len(frame), RAW_BARS, RAW_FEATURES)
    if not np.isfinite(raw).all():
        # допускаем нулевые ATR/BB в редких «пустых» барах -> заменяем на 0 не ломая схему
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Роу-локальная z-нормализация по окну для всех каналов, кроме direction.

    Считаем по всему окну (20 баров) mean/std (ddof=0) — как готовит RNN_Scaler/EA.
    Канал direction (индекс 4) остаётся {-1, +1} без нормировки.
    """
    normalized = features.astype(np.float64, copy=True)
    selected = np.asarray(NORMALIZE_MASK, dtype=np.int64)
    mean = normalized[:, :, selected].mean(axis=1, keepdims=True)
    std = normalized[:, :, selected].std(axis=1, ddof=0, keepdims=True)
    normalized[:, :, selected] = (normalized[:, :, selected] - mean) / np.maximum(std, EPS)
    if not np.isfinite(normalized).all():
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32)


def transform_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = raw_tensor(frame)                    # [N,20,12] raw
    model = normalize_features(raw)            # [N,20,12] normalized (direction raw)
    labels = frame["label"].apply(parse_label).to_numpy(dtype=np.int64)
    results = frame["result"].to_numpy(dtype=np.float64)
    timestamps = frame["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    return model, labels, results, timestamps


def quality_report(model: np.ndarray, labels: np.ndarray, source: str) -> dict:
    flat = model.reshape(-1, model.shape[-1])
    finite = np.isfinite(flat)
    return {
        "source": source,
        "rows": int(len(model)),
        "shape": list(model.shape),
        "model_features": list(MODEL_NAMES),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "zero_std_channels": [MODEL_NAMES[i] for i, v in enumerate(flat.std(axis=0)) if v < EPS],
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "global_min": float(np.min(flat[finite])) if finite.any() else None,
        "global_max": float(np.max(flat[finite])) if finite.any() else None,
        "schema_fingerprint": schema_fingerprint(),
    }


def process_file(path: Path, output_dir: Path) -> dict:
    frame = load_csv(path)
    model, labels, results, timestamps = transform_frame(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{path.stem}_normalized.npz"
    np.savez_compressed(target, X=model, y=labels, result=results, datetime=timestamps)
    report = quality_report(model, labels, path.name)
    report["output"] = target.name
    return report


def write_contract(output_dir: Path) -> None:
    contract = {
        "version": 3,
        "raw_bars": RAW_BARS,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "raw_names": list(RAW_NAMES),
        "model_names": list(MODEL_NAMES),
        "normalize_mask": list(NORMALIZE_MASK),
        "normalization": "row_local_zscore_ddof0_selected_channels",
        "label_map": {"[1,0]": 1, "[0,1]": 0, "else": 0},
        "eps": EPS,
        "schema_fingerprint": schema_fingerprint(),
        "onnx_input_shape": [1, RAW_BARS, MODEL_FEATURES],
    }
    (output_dir / "model_features.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare new BUY CSV files for Colab training")
    parser.add_argument("--input-dir", type=Path, default=Path(r"C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\Common\Files"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"H:\Мой диск\RNN\data_normalized"))
    parser.add_argument("--pattern", default="data_BUY_*.csv")
    args = parser.parse_args()
    paths = sorted(Path(p) for p in glob.glob(str(args.input_dir / args.pattern)))
    if not paths:
        raise SystemExit(f"No CSV files found in {args.input_dir} matching {args.pattern}")
    reports = [process_file(path, args.output_dir) for path in paths]
    write_contract(args.output_dir)
    (args.output_dir / "preparation_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for report in reports:
        print(f"{report['source']}: {report['rows']} rows -> {report['output']} | shape={report['shape']}")


if __name__ == "__main__":
    main()
