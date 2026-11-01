//+------------------------------------------------------------------+
//|                                PrepareDataOptimizationCriterion.mqh |
//|   Пользовательский критерий оптимизации для PrepareData.mq5       |
//|                                                                  |
//|   Считает метрики на основе статистики разметки датасета:       |
//|   - Win Rate: (TP + выше порога) / всего сделок                 |
//|   - Ratio: положительные / убыточные (по SL)                    |
//|   - Quality Score: взвешенный показатель качества разметки      |
//|                                                                  |
//|   Использование:                                                |
//|   - Переменные g_stat_tp, g_stat_thr, g_stat_sl, g_stat_other  |
//|     должны быть доступны (из PrepareData.mq5)                  |
//|   - Выберите "Пользовательский максимум" при оптимизации         |
//+------------------------------------------------------------------+
#ifndef PREPARE_DATA_OPTIMIZATION_CRITERION_MQH
#define PREPARE_DATA_OPTIMIZATION_CRITERION_MQH

//--- Конфигурация (внешние переменные из PrepareData.mq5) ---
extern int    g_stat_tp;      // [1,0]: закрыто строго по TP
extern int    g_stat_thr;     // [1,0]: закрыто выше порога (не по TP)
extern int    g_stat_sl;      // [0,1]: закрыто по SL
extern int    g_stat_other;   // [0,1]: прочее (горизонт/день)
extern int    buy_label0;     // всего [0,1]
extern int    buy_label1;     // всего [1,0]
extern int    rows_written;   // всего строк записано

//+------------------------------------------------------------------+
//| Структура статистики оптимизации                                 |
//+------------------------------------------------------------------+
struct PrepareDataOptimizationMetrics
  {
   int    total_trades;        // всего сделок (buy_label0 + buy_label1)
   int    positive_trades;     // положительные (g_stat_tp + g_stat_thr)
   int    sl_trades;           // закрыто по SL
   double win_rate;            // % положительных
   double sl_ratio;            // положительные / SL (ratio)
   double quality_score;       // комбинированная оценка качества
   double profit_factor;       // квазиPF: положительные / отрицательные
  };

//+------------------------------------------------------------------+
//| Расчёт метрик оптимизации                                        |
//+------------------------------------------------------------------+
void PrepareDataOptimization_Collect(PrepareDataOptimizationMetrics &metrics)
  {
   ZeroMemory(metrics);

   // --- Базовая статистика ---
   metrics.total_trades = buy_label0 + buy_label1;
   metrics.positive_trades = g_stat_tp + g_stat_thr;
   metrics.sl_trades = g_stat_sl;

   // --- Win Rate (%) ---
   if(metrics.total_trades > 0)
      metrics.win_rate = (100.0 * metrics.positive_trades) / metrics.total_trades;
   else
      metrics.win_rate = 0.0;

   // --- Ratio положительных к SL ---
   if(metrics.sl_trades > 0)
      metrics.sl_ratio = (double)metrics.positive_trades / metrics.sl_trades;
   else
      metrics.sl_ratio = (metrics.positive_trades > 0) ? 99.0 : 0.0;

   // --- Profit Factor (квази) ---
   int total_negative = g_stat_sl + g_stat_other;
   if(total_negative > 0)
      metrics.profit_factor = (double)metrics.positive_trades / total_negative;
   else
      metrics.profit_factor = (metrics.positive_trades > 0) ? 99.0 : 0.0;

   // --- Quality Score (комбинированная оценка) ---
   // Формула: (Win Rate * 0.5) + (SL Ratio * 0.5)
   // Нормализуем SL_Ratio к диапазону 0-100 (макс 100 = ratio >= 1.0)
   double normalized_sl_ratio = MathMin(metrics.sl_ratio * 100.0, 100.0);
   metrics.quality_score = (metrics.win_rate * 0.5) + (normalized_sl_ratio * 0.5);

   // --- Вывод в лог ---
   PrintFormat("PrepareDataOptimization: "
               "сделок=%d | положительные=%d (%.2f%%) | SL=%d | "
               "Ratio=%.2f | PF=%.2f | QualityScore=%.2f",
               metrics.total_trades,
               metrics.positive_trades, metrics.win_rate,
               metrics.sl_trades,
               metrics.sl_ratio, metrics.profit_factor,
               metrics.quality_score);
  }

//+------------------------------------------------------------------+
//| OnTester: возвращает критерий для оптимизации                    |
//+------------------------------------------------------------------+
double OnTester()
  {
   PrepareDataOptimizationMetrics metrics;
   PrepareDataOptimization_Collect(metrics);

   // Возвращаем Quality Score как основной критерий
   // (максимизируем комбинацию win_rate и sl_ratio)
   return metrics.quality_score;
  }

#endif // PREPARE_DATA_OPTIMIZATION_CRITERION_MQH
