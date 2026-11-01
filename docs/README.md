# RNN Gold M5 BUY

В корне лежат советник и вспомогательные MQL5-файлы, необходимые для его сборки и подготовки данных. Python-скрипты, ноутбуки и CSV сгруппированы в отдельных папках.

## Структура

- `RNNFilter_Gold_M5.mq5`, `RNNFilter_Gold_M5.ex5` — торговый советник.
- `PrepareData.mq5` и `PrepareData.ex5` — утилита подготовки данных.
- `include/` — все подключаемые модули `*.mqh`.
- `training/` — Python-пайплайн признаков и проверка/развёртывание модели.
- `notebooks/` — подбор параметров и финальное обучение RNN.
- `data/` — локальные CSV-выгрузки для обучения.

## Рабочий цикл

1. Подготовить CSV в MetaTrader с `PrepareData.mq5`.
2. Подготовить признаки: `python training/feature_pipeline.py --help`.
3. Настроить модель в `notebooks/rnn_otune.ipynb`, затем выполнить `notebooks/rnn_final_train_from_optuna.ipynb`.
4. Проверить и развернуть модель: `python training/deploy_rnn_model.py`.
