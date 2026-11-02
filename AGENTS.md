# Repository Guidelines

## Project Overview

This repository contains a MetaTrader 5 BUY-only Expert Advisor and its data and model preparation workflow. Keep changes focused on the requested behavior and preserve existing trading assumptions unless the task explicitly changes them.

## Project Structure

- `RNNFilter_Gold_M5.mq5` is the trading Expert Advisor.
- `PrepareData.mq5` exports labeled trade examples from MT5.
- `include/` contains shared MQL5 headers. Keep `.mqh` files here and use project-relative includes such as `#include "include/RNN_ONNX.mqh"`.
- `training/` contains Python feature preparation and model deployment scripts; `notebooks/` contains exploratory notebooks.
- `docs/` contains project documentation. `data/` is for local/runtime datasets; generated CSV data is ignored.

## Build and Development

Compile changed `.mq5` files in MetaEditor (F7), or use `metaeditor64.exe /compile:"<absolute-path>"` when available. Python training scripts are run with the project’s Python environment; inspect each script’s arguments before execution. There is no repository-wide build system or automated test suite. Do not claim compilation, backtests, or training completed unless actually run.

## Coding Conventions

Follow nearby MQL5 formatting and preserve platform event handler names. Prefix user inputs with `Inp`; use `.mqh` for shared headers and keep them in `include/`. Keep include paths relative to this project. In Python, follow the existing scripts’ style and avoid introducing a new dependency without need.

## Generated and Sensitive Files

`.gitignore` excludes compiled `.ex5` files, ONNX models, exported datasets, logs, and generated output. Keep these excluded unless a task explicitly requires a specific artifact. Never commit credentials, account details, or terminal logs.

## Change Review

Keep refactors separate from trading-logic changes where practical. Describe affected EA/data workflow and report the exact compile or runtime checks performed.

## MetaTrader MCP Scope

Use the built-in MetaEditor MCP only to compile MQL5 files. Do not use MCP terminal or other MetaTrader MCP operations for source inspection, workspace discovery, testing, trading, or runtime checks; inspect project files directly for code analysis.
