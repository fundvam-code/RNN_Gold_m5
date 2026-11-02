#!/usr/bin/env python3
"""Compare an EA diagnostic CSV against Python preprocessing and the same ONNX model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from feature_pipeline import EPS, RAW_BARS, RAW_FEATURES, RAW_NAMES, normalize_features

CONTRACT = "oldest_to_newest_last_seq_v1"
TENSOR_ATOL = 1e-5
TENSOR_RTOL = 1e-5
PROB_ATOL = 1e-5
PROB_RTOL = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_diagnostic(path: Path):
    meta, raw, tensor = {}, {}, {}
    with path.open(newline='', encoding='ascii') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            kind = row[0]
            if kind == 'meta' and len(row) == 3:
                if row[1] in meta:
                    raise ValueError(f'Duplicate metadata: {row[1]}')
                meta[row[1]] = row[2]
            elif kind == 'raw' and len(row) == 3 + RAW_FEATURES:
                index = int(row[1])
                if index in raw:
                    raise ValueError(f'Duplicate raw bar: {index}')
                raw[index] = (int(row[2]), [float(x) for x in row[3:]])
            elif kind == 'tensor' and len(row) == 2 + RAW_FEATURES:
                index = int(row[1])
                if index in tensor:
                    raise ValueError(f'Duplicate tensor bar: {index}')
                tensor[index] = [float(x) for x in row[2:]]
            else:
                raise ValueError(f'Invalid diagnostic row: {row}')
    required = {'version', 'model_sha256', 'signal_time', 'seq_len', 'profit_class', 'probability'}
    if not required.issubset(meta) or meta['version'] != '1':
        raise ValueError(f'Missing or unsupported metadata: {sorted(required - meta.keys())}')
    seq_len = int(meta['seq_len'])
    if not 1 <= seq_len <= RAW_BARS or set(raw) != set(range(RAW_BARS)) or set(tensor) != set(range(seq_len)):
        raise ValueError('Diagnostic window or sequence indices are invalid')
    times = np.array([raw[i][0] for i in range(RAW_BARS)], dtype=np.int64)
    for index in range(1, RAW_BARS):
        if times[index] <= times[index - 1]:
            raise ValueError(f'Bar time mismatch at {index}: previous={times[index-1]}, current={times[index]}')
    if times[-1] != int(meta['signal_time']):
        raise ValueError(f'Last bar time mismatch: bar={times[-1]}, signal={meta["signal_time"]}')
    raw_values = np.array([raw[i][1] for i in range(RAW_BARS)], dtype=np.float64)
    actual = np.array([tensor[i] for i in range(seq_len)], dtype=np.float32)
    if not np.isfinite(raw_values).all() or not np.isfinite(actual).all():
        raise ValueError('Non-finite raw or tensor value')
    directions = np.where(raw_values[:, 3] > raw_values[:, 0], 1.0, -1.0)
    for index in range(RAW_BARS):
        if raw_values[index, 4] != directions[index]:
            raise ValueError(f'Direction mismatch: bar {index}, expected={directions[index]}, EA={raw_values[index, 4]}')
    return meta, raw_values, actual


def validate(diagnostic: Path, model: Path, scaler: Path) -> dict:
    meta, raw, actual = read_diagnostic(diagnostic)
    model_hash = sha256(model)
    if meta['model_sha256'].lower() != model_hash:
        raise ValueError('The EA loaded a different ONNX model (SHA-256 mismatch)')
    seq_len = int(meta['seq_len'])
    expected = normalize_features(raw[None, :, :])[0, -seq_len:, :]
    mismatch = ~np.isclose(expected, actual, atol=TENSOR_ATOL, rtol=TENSOR_RTOL)
    if mismatch.any():
        bar, feature = np.argwhere(mismatch)[0]
        raise ValueError(f'Tensor mismatch: bar {bar} (window {RAW_BARS-seq_len+bar}), '
                         f'feature {RAW_NAMES[feature]}, Python={expected[bar, feature]:.9g}, '
                         f'EA={actual[bar, feature]:.9g}')
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0].type != 'tensor(float)':
        raise ValueError('ONNX must have one float32 input and one output')
    shape = inputs[0].shape
    if len(shape) != 3 or shape[1:] != [seq_len, RAW_FEATURES] or (isinstance(shape[0], int) and shape[0] != 1):
        raise ValueError(f'ONNX input shape mismatch: {shape}')
    logits = np.asarray(session.run(None, {inputs[0].name: expected[None, :, :]})[0], dtype=np.float64)
    if logits.shape != (1, 2) or not np.isfinite(logits).all():
        raise ValueError(f'Invalid ONNX output: {logits}')
    class_id = int(meta['profit_class'])
    if class_id not in (0, 1):
        raise ValueError('Invalid profit class')
    exp = np.exp(logits[0] - logits[0].max())
    probability = float(exp[class_id] / exp.sum())
    actual_probability = float(meta['probability'])
    if not np.isfinite(actual_probability) or not np.isclose(probability, actual_probability, atol=PROB_ATOL, rtol=PROB_RTOL):
        raise ValueError(f'Probability mismatch: Python={probability:.9g}, EA={actual_probability:.9g}')
    return {'passed': True, 'input_contract': CONTRACT, 'model_sha256': model_hash,
            'scaler_sha256': sha256(scaler), 'diagnostic_sha256': sha256(diagnostic),
            'seq_len': seq_len, 'raw_bars': RAW_BARS, 'features': RAW_FEATURES,
            'eps': EPS, 'signal_time': int(meta['signal_time']),
            'max_tensor_abs_diff': float(np.max(np.abs(expected - actual))),
            'python_probability': probability, 'ea_probability': actual_probability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scaler', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.unlink(missing_ok=True)  # a failed check must not leave a successful old report
    report = validate(args.diagnostic, args.model, args.scaler)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"EA/ONNX parity passed: {args.report}; max tensor error={report['max_tensor_abs_diff']:.3g}")


if __name__ == '__main__':
    main()
