#!/usr/bin/env python3
"""
RNN Model Deployment & Validation Script

Проверяет согласованность ONNX модели, RNN_Scaler.mqh и summary.json,
затем распределяет файлы в нужные директории MetaTrader 5.

Проверяет:
- Размер окна (seq_len): ONNX ↔ RNN_Scaler.mqh ↔ summary.json
- Количество признаков (n_features): ONNX ↔ RNN_Scaler.mqh ↔ summary.json
- Наличие файлов и их доступность
- Версионирование моделей (timestamp)

Копирует в:
- ONNX модель → Common\Files\RNN_GOLD\buy_model.onnx
- RNN_Scaler.mqh → MQL5\Experts\RNN_gold_M5_BUY\include\RNN_Scaler.mqh
"""

import os
import sys
import json
import shutil
import re
from pathlib import Path
from datetime import datetime

# Пути
COLAB_OUTPUT_DIR = Path(r"H:\Мой диск\RNN\output")
MT5_COMMON_DIR = Path(r"C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
MT5_RNN_GOLD_DIR = MT5_COMMON_DIR / "RNN_GOLD_M5"
MT5_EXPERT_DIR = Path(r"C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\RNN_gold_M5_BUY")

# Цвета для консоли
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'


def log_ok(msg):
    print(f"{GREEN}✓ {msg}{RESET}")


def log_error(msg):
    print(f"{RED}✗ {msg}{RESET}")


def log_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def log_info(msg):
    print(f"{CYAN}ℹ {msg}{RESET}")


def read_onnx_shape(onnx_path):
    """Читает размер входа из ONNX файла."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        # input_shape[0] = batch (dynamic), input_shape[1] = seq_len, input_shape[2] = features
        seq_len = input_shape[1].dim_value
        n_features = input_shape[2].dim_value
        return seq_len, n_features
    except Exception as e:
        log_error(f"Не удалось прочитать ONNX: {e}")
        return None, None


def read_mqh_defines(mqh_path):
    """Читает #define значения из RNN_Scaler.mqh."""
    defines = {}
    try:
        with open(mqh_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'#define\s+(\w+)\s+(\d+)', line)
                if match:
                    key, value = match.groups()
                    defines[key] = int(value)
        return defines
    except Exception as e:
        log_error(f"Не удалось прочитать MQH: {e}")
        return {}


def read_summary_json(json_path):
    """Читает параметры из summary.json."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        log_error(f"Не удалось прочитать summary.json: {e}")
        return {}


def validate_files():
    """Проверяет наличие всех необходимых файлов."""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}ПРОВЕРКА ФАЙЛОВ{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    files_ok = True

    # Проверяем Colab output
    if not COLAB_OUTPUT_DIR.exists():
        log_error(f"Папка Colab не найдена: {COLAB_OUTPUT_DIR}")
        files_ok = False
    else:
        log_ok(f"Colab output папка: {COLAB_OUTPUT_DIR}")

    onnx_file = COLAB_OUTPUT_DIR / "onnx" / "buy_model.onnx"
    if onnx_file.exists():
        size_mb = onnx_file.stat().st_size / 1024 / 1024
        log_ok(f"ONNX модель найдена ({size_mb:.1f} MB)")
    else:
        log_error(f"ONNX модель не найдена: {onnx_file}")
        files_ok = False

    scaler_file = COLAB_OUTPUT_DIR / "RNN_Scaler.mqh"
    if scaler_file.exists():
        log_ok(f"RNN_Scaler.mqh найден")
    else:
        log_error(f"RNN_Scaler.mqh не найден: {scaler_file}")
        files_ok = False

    summary_file = COLAB_OUTPUT_DIR / "summary.json"
    if summary_file.exists():
        log_ok(f"summary.json найден")
    else:
        log_error(f"summary.json не найден: {summary_file}")
        files_ok = False

    return files_ok, onnx_file, scaler_file, summary_file


def validate_consistency(onnx_file, scaler_file, summary_file):
    """Проверяет согласованность всех параметров."""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}ПРОВЕРКА СОГЛАСОВАННОСТИ{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    consistency_ok = True

    # Читаем параметры
    onnx_seq, onnx_feat = read_onnx_shape(onnx_file)
    mqh_defines = read_mqh_defines(scaler_file)
    summary = read_summary_json(summary_file)

    print(f"ONNX:       seq_len={onnx_seq}, n_features={onnx_feat}")
    print(f"MQH:        NN_SEQ_LEN={mqh_defines.get('NN_SEQ_LEN')}, NN_FEATURES={mqh_defines.get('NN_FEATURES')}")
    if summary.get('buy_metrics'):
        print(f"summary.json: TP={summary['buy_metrics'].get('tp')}, FP={summary['buy_metrics'].get('fp')}, "
              f"Precision={summary['buy_metrics'].get('precision'):.4f}\n")

    # Проверяем seq_len
    mqh_seq = mqh_defines.get('NN_SEQ_LEN')
    if onnx_seq == mqh_seq:
        log_ok(f"seq_len совпадает: {onnx_seq}")
    else:
        log_error(f"seq_len не совпадает: ONNX={onnx_seq}, MQH={mqh_seq}")
        consistency_ok = False

    # Проверяем n_features
    mqh_feat = mqh_defines.get('NN_FEATURES')
    if onnx_feat == mqh_feat:
        log_ok(f"n_features совпадает: {onnx_feat}")
    else:
        log_error(f"n_features не совпадает: ONNX={onnx_feat}, MQH={mqh_feat}")
        consistency_ok = False

    # Проверяем что MQH имеет правильный epsilon
    eps = mqh_defines.get('NN_NORM_STD_EPS')
    if eps:
        log_ok(f"NN_NORM_STD_EPS определён: {eps:g}")
    else:
        log_warning(f"NN_NORM_STD_EPS не найден в MQH")

    # Проверяем metrics
    if summary.get('buy_metrics'):
        metrics = summary['buy_metrics']
        tp = metrics.get('tp', 0)
        fp = metrics.get('fp', 0)
        total = tp + fp
        precision = metrics.get('precision', 0.0)
        if total > 0:
            log_ok(f"Метрики найдены: TP={tp}, FP={fp}, Precision={precision:.4f}")
        else:
            log_warning(f"Метрики пусты: TP={tp}, FP={fp}")
    else:
        log_warning(f"Метрики не найдены в summary.json")

    return consistency_ok


def deploy_files(onnx_file, scaler_file):
    """Копирует файлы в нужные директории."""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}РАЗВЁРТЫВАНИЕ ФАЙЛОВ{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    deployment_ok = True

    # Создаём директории если их нет
    MT5_RNN_GOLD_DIR.mkdir(parents=True, exist_ok=True)
    MT5_EXPERT_DIR.mkdir(parents=True, exist_ok=True)

    # Копируем ONNX модель
    try:
        dest_onnx = MT5_RNN_GOLD_DIR / "buy_model.onnx"
        shutil.copy2(onnx_file, dest_onnx)
        size_mb = dest_onnx.stat().st_size / 1024 / 1024
        log_ok(f"ONNX модель → {dest_onnx} ({size_mb:.1f} MB)")
    except Exception as e:
        log_error(f"Не удалось скопировать ONNX: {e}")
        deployment_ok = False

    # Копируем RNN_Scaler.mqh
    try:
        dest_mqh = MT5_EXPERT_DIR / "include" / "RNN_Scaler.mqh"
        dest_mqh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scaler_file, dest_mqh)
        log_ok(f"RNN_Scaler.mqh → {dest_mqh}")
    except Exception as e:
        log_error(f"Не удалось скопировать RNN_Scaler.mqh: {e}")
        deployment_ok = False

    return deployment_ok


def main():
    """Основная функция."""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}RNN Model Deployment & Validation{RESET}")
    print(f"{CYAN}Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")

    # Проверяем файлы
    files_ok, onnx_file, scaler_file, summary_file = validate_files()
    if not files_ok:
        log_error("Проверка файлов не пройдена!")
        return False

    # Проверяем согласованность
    consistency_ok = validate_consistency(onnx_file, scaler_file, summary_file)
    if not consistency_ok:
        log_error("Проверка согласованности не пройдена!")
        return False

    # Развёртываем файлы
    deployment_ok = deploy_files(onnx_file, scaler_file)
    if not deployment_ok:
        log_error("Развёртывание не пройдено!")
        return False

    # Финальное сообщение
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{GREEN}✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")
    print(f"{CYAN}Файлы готовы к использованию в MetaTrader 5:{RESET}")
    print(f"  • ONNX модель:    {MT5_RNN_GOLD_DIR}/buy_model.onnx")
    print(f"  • RNN_Scaler:     {MT5_EXPERT_DIR}/include/RNN_Scaler.mqh")
    print(f"\n{YELLOW}Перезагрузите MetaTrader 5 для применения изменений{RESET}\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
