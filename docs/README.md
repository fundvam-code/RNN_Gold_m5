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
4. Проверить и развернуть модель по инструкции ниже.

## Порядок входных данных и выпуск модели

В CSV lag0 — самый старый, а lag19 — ближайший к сигналу бар M5. Все 20 баров используются для нормализации; GRU получает последние seq_len баров от старого к свежему. После изменения контракта требуется заново выполнить Optuna и обучение. Старые модели несовместимы.

1. Запустите `training/feature_pipeline.py`, затем ноутбуки `rnn_otune.ipynb` и `rnn_final_train_from_optuna.ipynb`. Финальный ноутбук сравнит PyTorch и ONNX и создаст `final_results.json` только после успешной проверки. Для проверки нужны `onnx` и `onnxruntime`.
2. Подготовьте кандидата: `python training/deploy_rnn_model.py --stage`. Скрипт положит `buy_model.candidate.onnx` в `Common/Files/RNN_GOLD_M5` и скопирует заголовок в `include`. Каталоги можно изменить аргументами `--output-dir`, `--common-dir` и `--project-dir`.
3. Скомпилируйте `RNNFilter_Gold_M5.mq5` в MetaEditor. В тестере стратегий задайте `Inp_NNDiagnostics=true`, `Inp_UseNNFilter=true` и `Inp_BuyModelFile=RNN_GOLD_M5\buy_model.candidate.onnx`. Режим не совершает и не закрывает сделки; после подготовки истории он запишет `Common/Files/RNN_GOLD_M5/diagnostic.csv`. Последний диагностический бар — последний закрытый M5 бар в момент снимка.
4. В PowerShell из папки проекта проверьте снимок и затем опубликуйте кандидата:

```powershell
$common = 'C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\Common\Files\RNN_GOLD_M5'
$diag = Join-Path $common 'diagnostic.csv'
$model = Join-Path $common 'buy_model.candidate.onnx'
$report = Join-Path $common 'diagnostic_report.json'
python training/verify_rnn_diagnostic.py --diagnostic $diag --model $model --scaler include/RNN_Scaler.mqh --report $report
python training/deploy_rnn_model.py --finalize --diagnostic $diag --report $report
```

Проверка сравнивает SHA-256 модели, времена баров, исходные значения признаков, нормализованный тензор и вероятность EA с Python/ONNX. При расхождении указывается первый бар и признак либо вероятность. Скрипт развёртывания повторяет проверку и публикует `buy_model.onnx` только при совпадении отчёта, модели и заголовка. После публикации в обычном режиме укажите стандартный путь `RNN_GOLD_M5\buy_model.onnx` и выключите диагностику.

До нового обучения SHA-256 модели в заголовке проекта пуст, поэтому загрузка BUY ONNX запрещена. Заголовок после `--stage` связан с конкретным файлом модели; при каждой новой модели нужна повторная компиляция EA.
