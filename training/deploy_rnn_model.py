#!/usr/bin/env python3
"""Stage a trained model, then publish it only after EA diagnostics pass.

Use --stage before compiling and running the EA in the tester. Run
verify_rnn_diagnostic.py on the resulting CSV, then use --finalize.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

CONTRACT = 'oldest_to_newest_last_seq_v1'
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(r'H:\Мой диск\RNN\output')
DEFAULT_COMMON = Path(r'C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\RNN_GOLD_M5')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_defines(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*#define\s+(NN_\w+)\s+([^\s/]+)', line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def validate_export(output: Path):
    import onnx
    model = output / 'onnx' / 'buy_model.onnx'
    scaler = output / 'RNN_Scaler.mqh'
    summary = json.loads((output / 'final_results.json').read_text(encoding='utf-8'))
    if summary.get('input_contract') != CONTRACT or summary.get('model_sha256') != sha256(model):
        raise ValueError('Training metadata does not match model or current input contract')
    parity = summary.get('onnx_parity', {})
    if parity.get('passed') is not True or parity.get('samples', 0) < 1:
        raise ValueError('PyTorch/ONNX comparison has not passed')
    config = summary['model_config']
    seq, features, window = int(config['seq_len']), int(config['n_features']), int(config['n_bars'])
    if not 1 <= seq <= window or window != 20 or features != 12:
        raise ValueError('Invalid training model configuration')
    defs = read_defines(scaler)
    if defs.get('NN_BUY_MODEL_SHA256', '').strip('\"') != sha256(model):
        raise ValueError('Scaler model hash does not match ONNX')
    expected = {'NN_BUY_SEQ_LEN': seq, 'NN_SELL_SEQ_LEN': seq, 'NN_MAX_SEQ_LEN': seq,
                'NN_FEATURES': features, 'NN_BUY_NORM_WINDOW': window,
                'NN_SELL_NORM_WINDOW': window, 'NN_MAX_NORM_WINDOW': window}
    for key, value in expected.items():
        if defs.get(key) != str(value):
            raise ValueError(f'Scaler {key}: expected {value}, got {defs.get(key)}')
    if float(defs.get('NN_NORM_STD_EPS', 'nan')) != 1e-6:
        raise ValueError('Scaler epsilon must be 1e-6')
    graph = onnx.load(str(model), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError('ONNX external data is unsupported')
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError('ONNX must have one input and one output')
    tensor = graph.graph.input[0].type.tensor_type
    dims = tensor.shape.dim
    if (tensor.elem_type != onnx.TensorProto.FLOAT or len(dims) != 3 or
        (dims[0].HasField('dim_value') and dims[0].dim_value != 1) or
        dims[1].dim_value != seq or dims[2].dim_value != features):
        raise ValueError('ONNX input must be float32 [1 or batch, seq_len, 12]')
    return model, scaler, summary


def copy_atomic(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--stage', action='store_true')
    action.add_argument('--finalize', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--common-dir', type=Path, default=DEFAULT_COMMON)
    parser.add_argument('--project-dir', type=Path, default=PROJECT_DIR)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    model, scaler, summary = validate_export(args.output_dir)
    staged_model = args.common_dir / 'buy_model.candidate.onnx'
    staged_scaler = args.project_dir / 'include' / 'RNN_Scaler.mqh'
    if args.stage:
        copy_atomic(model, staged_model)
        copy_atomic(scaler, staged_scaler)
        print(f'Staged model: {staged_model}')
        print(f'Staged scaler: {staged_scaler}')
        print('Compile the EA, then run tester diagnostics using the candidate model.')
        return
    if args.diagnostic is None or args.report is None:
        parser.error('--finalize requires --diagnostic and --report')
    if sha256(staged_model) != sha256(model) or sha256(staged_scaler) != sha256(scaler):
        raise ValueError('Staged model or EA scaler differs from training output; stage and test again')
    from verify_rnn_diagnostic import validate
    actual = validate(args.diagnostic, staged_model, staged_scaler)
    saved = json.loads(args.report.read_text(encoding='utf-8'))
    identity_keys = ('passed', 'input_contract', 'model_sha256', 'scaler_sha256',
                     'diagnostic_sha256', 'seq_len', 'raw_bars', 'features', 'eps', 'signal_time')
    if (any(key not in saved or saved[key] != actual[key] for key in identity_keys) or
        actual['seq_len'] != summary['model_config']['seq_len']):
        raise ValueError('Diagnostic report does not match the staged model, scaler, and current diagnostic')
    copy_atomic(staged_model, args.common_dir / 'buy_model.onnx')
    print(f'Published verified model: {args.common_dir / "buy_model.onnx"}')


if __name__ == '__main__':
    main()
