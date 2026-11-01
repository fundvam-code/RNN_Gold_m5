//+------------------------------------------------------------------+
//|                                                  PrepareData.mq5 |
//|                                      Copyright 2026, Bondarev A. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Bondarev A."
#property link      "https://www.mql5.com"
#property version   "2.03"
//+------------------------------------------------------------------+
//|                    Экспорт BUY-сигналов и разметка в CSV         |
//|   Сигнал ТОЛЬКО на покупку (ТФ сигнала = Inp_BBTimeframe):      |
//|     1) свеча ТФ сигнала закрылась НИЖЕ нижней полосы Боллинджера;|
//|     2) следующая свеча закрылась ВЫШЕ сигнальной (разворот);    |
//|     3) тренд H4 (ADX/+DI/-DI) = BULL;                           |
//|     4) вход рыночным BUY на открытии следующей свечи (ТФ сигнала).|
//|   Тренд-фильтр H4: BULL/BEAR/FLAT (метка с цветом на графике).  |
//|   Признаки (окно 20 баров M5, лаг0 = сигнальный бар):           |
//|     open, high, low, close (без объёма), direction, ema8,       |
//|     ema21, rsi, atr, bb_upper, bb_mid, bb_lower.                |
//|   Горизонт удержания и проверка конца дня считаются в M1,       |
//|   независимо от Inp_BBTimeframe; экспорт в Common\Files:        |
//|   data_BUY_*.csv.                                                |
//|   label: [1,0] - закрытие по TP ИЛИ результат >= порога в у.е.  |
//|   (Inp_ProfitThresholdUSD); [0,1] - всё остальное (SL, горизонт, |
//|   конец дня, OnDeinit). Рядом с CSV пишется отчёт <CSV>_stats.txt:
//|   всего меток, [1,0]/[0,1], разбивка [1,0] на TP / выше порога. |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Files\FileTxt.mqh>
#include "include/H4_trend_filter.mqh"

#define EXPORT_SEQ_LEN   20
#define EXPORT_FEATURES  12
#define SIGNAL_PERIOD    PERIOD_M1   // цикл OnTick, горизонт удержания, конец дня - всегда M1
// ТФ отскока/закрытия BB-свечей - входной параметр Inp_BBTimeframe (детекция сигнала)
#define FEATURE_PERIOD   PERIOD_M5
#define TREND_PERIOD     PERIOD_H4

// --- Входные параметры ---
// Каждая строка CSV содержит окно из 20 баров M5: от сигнального к более старым.
input double   Inp_LotSize      = 0.01;      // Объём ордера (лоты)
input int      Inp_Magic        = 777001;    // Магик-номер
input double   Inp_TP_Points    = 1000.0;    // Тейк-профит (пункты), 0 - без TP
input double   Inp_SL_Points    = 1000.0;    // Стоп-лосс (пункты), 0 - без SL
input int      Inp_HoldBars     = 20;        // Макс. горизонт удержания (бары M1)
input double   Inp_ProfitThresholdUSD = 5.0; // Порог прибыли (у.е.) для метки [1,0], 0 = только TP
input ENUM_TIMEFRAMES Inp_BBTimeframe = PERIOD_M1;  // Таймфрейм сигнала: BB + закрытие свечей
input int      Inp_BB_Period    = 20;        // Боллинджер: период
input double   Inp_BB_Dev       = 2.0;       // Боллинджер: множитель (сигма)
input int      Inp_ADX_Period   = 14;        // ADX/+DI/-DI: период
input double   Inp_ADX_Threshold= 25.0;      // ADX: порог тренда (классика)

// --- Режим работы ---
input bool     Inp_ExportData   = true;      // Экспортировать данные в CSV и отчеты (отключить при оптимизации)

// --- Пользовательский критерий оптимизации (разметка датасета) ---
#include "include/PrepareDataOptimizationCriterion.mqh"

// --- Глобальные переменные ---
// Причина выхода из сделки (для разметки и статистики)
enum ENUM_EXIT_KIND
  {
   EXIT_TP = 0,   // закрытие по TP
   EXIT_SL = 1,   // закрытие по SL
   EXIT_EA = 2    // закрыто советником (горизонт / конец дня / OnDeinit)
  };

CTrade         trade;
int            h_ema8, h_ema21, h_rsi, h_atr, h_bb;    // M5: признаки + график
int            h_bb_sig;                               // ТФ сигнала: сигнальный Боллинджер
CFileTxt       csvFileBuy;                             // BUY-файл (только BUY)
bool           files_ok = false;
int            rows_written = 0;
int            buy_signals = 0;                        // сигналов (пошло в торговлю)
int            buy_trades = 0;                         // фактически открытых сделок
int            buy_label0 = 0, buy_label1 = 0;         // записано строк: label=[0,1] / [1,0]
// --- Статистика разметки (пишется в <имя CSV>_stats.txt) ---
int            g_stat_tp    = 0;   // [1,0]: закрыто строго по TP
int            g_stat_thr   = 0;   // [1,0]: закрыто не по TP, но результат >= порога (у.е.)
int            g_stat_sl    = 0;   // [0,1]: закрыто по SL
int            g_stat_other = 0;   // [0,1]: закрыто иначе (горизонт / конец дня / OnDeinit)
double         tp_price = 0.0, sl_price = 0.0;
int            total_features = 0;                     // 20 * EXPORT_FEATURES

// --- Отслеживание открытых сделок ---
struct DealRecord
  {
   ulong         ticket;        // тикет позиции
   datetime      signal_time;   // время свечи подтверждения M1
   int           held_bars;     // сколько баров M1 позиция уже открыта
   double        features[];    // признаковый вектор
  };
DealRecord     deals[];
int            deals_count = 0;
datetime       last_bar_time = 0;
datetime       test_start = 0;                          // первый бар теста
datetime       test_end = 0;                            // последний бар теста
string         fnameBuy = "";                           // текущее имя BUY-файла (временное)
datetime       g_last_bb_bar = 0;                       // последний обработанный бар Inp_BBTimeframe (защита от повторов на M1)
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!MQLInfoInteger(MQL_TESTER))
     {
      Print("Советник предназначен только для тестера стратегий!");
      return(INIT_FAILED);
     }

   if(Inp_HoldBars < 1 || Inp_LotSize <= 0 ||
      Inp_TP_Points < 0.0 || Inp_SL_Points < 0.0 ||
      Inp_ProfitThresholdUSD < 0.0 || Inp_BBTimeframe == PERIOD_CURRENT ||
      Inp_BB_Period < 2 || Inp_BB_Dev <= 0.0 ||
      Inp_ADX_Period < 2 || Inp_ADX_Threshold <= 0.0 || Inp_ADX_Threshold >= 100.0)
     {
      Print("Проверьте параметры: HoldBars>=1, LotSize>0, TP/SL>=0, ProfitThresholdUSD>=0 (0=только TP), BB-TF != Current, BB(period>=2, dev>0), ADX(2..period, порог 0..100)");
      return(INIT_FAILED);
     }

   // 1 пункт = 1 шаг цены (минимум изменения)
   tp_price = Inp_TP_Points * _Point;
   sl_price = Inp_SL_Points * _Point;

   total_features = EXPORT_SEQ_LEN * EXPORT_FEATURES;

   // --- M5: индикаторы признаков (они же выводятся на график) ---
   h_ema8 = iMA(_Symbol, FEATURE_PERIOD, 8,  0, MODE_EMA, PRICE_CLOSE);
   h_ema21= iMA(_Symbol, FEATURE_PERIOD, 21, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi  = iRSI(_Symbol, FEATURE_PERIOD, 14, PRICE_CLOSE);
   h_atr  = iATR(_Symbol, FEATURE_PERIOD, 14);
   h_bb   = iBands(_Symbol, FEATURE_PERIOD, Inp_BB_Period, 0, Inp_BB_Dev, PRICE_CLOSE);

   // --- ТФ сигнала: сигнальный Боллинджер (отскок и подтверждение считаются на этом ТФ) ---
   h_bb_sig = iBands(_Symbol, Inp_BBTimeframe, Inp_BB_Period, 0, Inp_BB_Dev, PRICE_CLOSE);

   // --- Инициализация фильтра тренда H4 ---
   if(!H4_TrendFilter_Init(Inp_ADX_Period, Inp_ADX_Threshold))
      return(INIT_FAILED);

   int handles[6] = {h_ema8, h_ema21, h_rsi, h_atr, h_bb, h_bb_sig};
   for(int i = 0; i < 6; i++)
     {
      if(handles[i] == INVALID_HANDLE)
        {
         PrintFormat("Ошибка создания индикатора %d (код: %d)", i, GetLastError());
         return(INIT_FAILED);
        }
     }

   trade.SetExpertMagicNumber(Inp_Magic);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFillingBySymbol(_Symbol);

   // --- Визуализация: стандартные индикаторы на график тестера ---
   // Боллинджер и RSI (M5) — признаки и отображение; Боллинджер M1 — сигнал.
   if(ChartPeriod() == FEATURE_PERIOD)
     {
      if(ChartIndicatorAdd(0, 0, h_bb))
         Print("BB(M5) добавлен на график");
      if(ChartIndicatorAdd(0, 1, h_rsi))
         Print("RSI(M5) добавлен на график");
     }
   else
      PrintFormat("График тестера на %s - BB/RSI(M5) не добавлены (визуализация пропущена)",
                  EnumToString(ChartPeriod()));

   H4_TrendFilter_Update(Inp_ADX_Threshold);   // сразу обновляем состояние тренда

   PrintFormat("PrepareData v2.03 инициализирован: сигнал %s (BB %d/%.1f), тренд H4 (ADX %d, порог %.0f), признаки M5 %dx%d",
               EnumToString(Inp_BBTimeframe), Inp_BB_Period, Inp_BB_Dev, Inp_ADX_Period, Inp_ADX_Threshold,
               EXPORT_SEQ_LEN, EXPORT_FEATURES);
   PrintFormat("Удержание: %d баров %s, порог прибыли для метки [1,0]: %.2f у.е. (0 = только TP)",
               Inp_HoldBars, EnumToString(SIGNAL_PERIOD), Inp_ProfitThresholdUSD);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Закрываем оставшиеся позиции и записываем их результат
   for(int i = deals_count - 1; i >= 0; i--)
     {
      double result = 0.0;
      ENUM_EXIT_KIND kind = EXIT_EA;
      if(PositionSelectByTicket(deals[i].ticket))
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         trade.PositionClose(deals[i].ticket);   // принудительное закрытие (конец теста)
         result = profit;
         kind   = EXIT_EA;
        }
      else
        {
         // позиция уже закрыта (TP/SL) - результат и причина по истории сделок
         double p = 0.0;
         ENUM_EXIT_KIND k = EXIT_EA;
         GetClosedDealExit(deals[i].ticket, p, k);
         result = p;
         kind   = k;
        }
      WriteDealRow(deals[i].signal_time, deals[i].features, result, kind);
     }
   deals_count = 0;
   ArrayResize(deals, 0);

   int handles[6] = {h_ema8, h_ema21, h_rsi, h_atr, h_bb, h_bb_sig};
   for(int i = 0; i < 6; i++)
      if(handles[i] != INVALID_HANDLE)
         IndicatorRelease(handles[i]);

   // Деинициализация фильтра тренда H4
   H4_TrendFilter_Deinit();

   csvFileBuy.Close();

   // Переименовываем файл: имя должно отражать реальный интервал теста
   test_end = last_bar_time;
   if(test_end == 0)
      test_end = test_start;
   if(files_ok)
     {
      string new_buy = GenerateFileName("BUY", test_start, test_end);
      if(fnameBuy != "" && fnameBuy != new_buy)
        {
         if(FileIsExist(new_buy, FILE_COMMON))
            FileDelete(new_buy, FILE_COMMON);
         if(FileMove(fnameBuy, FILE_COMMON, new_buy, FILE_COMMON))
            fnameBuy = new_buy;
         else
            PrintFormat("PrepareData: не удалось переименовать '%s' -> '%s' (код %d)", fnameBuy, new_buy, GetLastError());
        }
     }

   PrintFormat("PrepareData v2.03: сигналов BUY=%d, сделок=%d, строк=%d ([1,0]: %d, [0,1]: %d)", buy_signals, buy_trades, rows_written, buy_label1, buy_label0);
   // Отчёт-статистика рядом с CSV (Common/Files): <имя CSV>_stats.txt
   if(files_ok && Inp_ExportData)
      WriteStatsReport(fnameBuy);
   int    total_rows = buy_label1 + buy_label0;              // всего меток (строк)
   double tp_pct = (buy_label1 > 0) ? (100.0 * g_stat_tp) / (double)buy_label1 : 0.0;
   PrintFormat("СТАТИСТИКА: меток=%d, [1,0]=%d, [0,1]=%d | [1,0]: строго TP=%.2f%%, выше порога=%d | [0,1]: SL=%d, прочее=%d",
               total_rows, buy_label1, buy_label0, tp_pct, g_stat_thr, g_stat_sl, g_stat_other);

  }
//+------------------------------------------------------------------+
//| Expert tick function: обрабатываем только на новой свече M1     |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime cur_bar = iTime(_Symbol, SIGNAL_PERIOD, 0);
   if(cur_bar == last_bar_time)
      return;
   last_bar_time = cur_bar;

   if(test_start == 0)
      test_start = cur_bar;

   if(!files_ok && Inp_ExportData)
      EnsureFilesOpen();

   // Обновляем состояние тренда H4
   H4_TrendFilter_Update(Inp_ADX_Threshold);
   H4_TrendFilter_DrawLabel();

   // На последнем баре M1 дня закрываем все позиции до конца дня
   if(IsLastBarOfDay(cur_bar, SIGNAL_PERIOD))
     {
      CloseAllDeals();
      return;   // новые сигналы в конце дня не открываем
     }

   ProcessOpenDeals();   // закрытие позиций по горизонту / обработка TP/SL
   DetectAndTrade();     // новые сигналы (только BUY)
  }
//+------------------------------------------------------------------+
//| Признак последнего бара торгового дня для заданного периода     |
//+------------------------------------------------------------------+
bool IsLastBarOfDay(datetime bar_time, ENUM_TIMEFRAMES period)
  {
   MqlDateTime dt, dt_next;
   TimeToStruct(bar_time, dt);
   TimeToStruct(bar_time + PeriodSeconds(period), dt_next);
   return(dt.day != dt_next.day);
  }
//+------------------------------------------------------------------+
//| Принудительное закрытие всех позиций до конца дня               |
//+------------------------------------------------------------------+
void CloseAllDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      double result = 0.0;
      ENUM_EXIT_KIND kind = EXIT_EA;
      if(PositionSelectByTicket(deals[i].ticket))
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         trade.PositionClose(deals[i].ticket);   // закрытие до конца дня
         result = profit;
         kind   = EXIT_EA;
        }
      else
        {
         // позиция уже закрыта (TP/SL) - результат и причина по истории сделок
         double p = 0.0;
         ENUM_EXIT_KIND k = EXIT_EA;
         GetClosedDealExit(deals[i].ticket, p, k);
         result = p;
         kind   = k;
        }
      WriteDealRow(deals[i].signal_time, deals[i].features, result, kind);
      RemoveDeal(i);
     }
  }
//+------------------------------------------------------------------+
//| Результат и причина закрытия позиции по истории сделок.         |
//| Причина берётся из DEAL_REASON: TP/SL - реальный ордер сработал,|
//| иначе (CLIENT/EXPERT/SO и т.п.) - закрытие инициировано EA.     |
//+------------------------------------------------------------------+
bool GetClosedDealExit(ulong position_ticket, double &profit, ENUM_EXIT_KIND &kind)
  {
   kind = EXIT_EA;
   if(!HistorySelectByPosition(position_ticket))
      return(false);

   int total = HistoryDealsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP);
      long reason = HistoryDealGetInteger(ticket, DEAL_REASON);
      if(reason == DEAL_REASON_TP)
         kind = EXIT_TP;
      else if(reason == DEAL_REASON_SL)
         kind = EXIT_SL;
      else
         kind = EXIT_EA;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Формирование имени файла по интервалу теста                      |
//+------------------------------------------------------------------+
string GenerateFileName(string signal_type, datetime start, datetime end)
  {
   MqlDateTime s, e;
   TimeToStruct(start, s);
   TimeToStruct(end, e);
   string start_str = StringFormat("%04d%02d%02d", s.year, s.mon, s.day);
   string end_str   = StringFormat("%04d%02d%02d", e.year, e.mon, e.day);
   return StringFormat("data_%s_%s_%s.csv", signal_type, start_str, end_str);
  }
//+------------------------------------------------------------------+
//| Открытие CSV-файла (Common\Files) на первой свече               |
//| Имя временное (start_start); финальное (start_end) формируется   |
//| в OnDeinit через FileMove.                                       |
//+------------------------------------------------------------------+
void EnsureFilesOpen()
  {
   if(test_start == 0)
      test_start = iTime(_Symbol, SIGNAL_PERIOD, 1);

   fnameBuy = GenerateFileName("BUY", test_start, test_start);
   if(!csvFileBuy.Open(fnameBuy, FILE_CSV|FILE_WRITE|FILE_ANSI|FILE_COMMON))
     {
      Print("Ошибка открытия файла: ", fnameBuy);
      return;
     }
   WriteHeader(csvFileBuy);
   files_ok = true;
  }
//+------------------------------------------------------------------+
//| Закрытие позиций: обработка TP/SL и горизонта Inp_HoldBars (M1) |
//+------------------------------------------------------------------+
void ProcessOpenDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         // позиция закрыта сервером (TP/SL) - результат и причина по истории сделок
         double p = 0.0;
         ENUM_EXIT_KIND kind = EXIT_EA;
         GetClosedDealExit(deals[i].ticket, p, kind);
         WriteDealRow(deals[i].signal_time, deals[i].features, p, kind);
         RemoveDeal(i);
         continue;
        }

      deals[i].held_bars++;
      if(deals[i].held_bars >= Inp_HoldBars)
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         trade.PositionClose(deals[i].ticket);   // закрытие по горизонту удержания
         WriteDealRow(deals[i].signal_time, deals[i].features, profit, EXIT_EA);
         RemoveDeal(i);
        }
     }
  }
//+------------------------------------------------------------------+
//| Детекция BUY-сигнала и открытие ордера                           |
//| Сигнал (ТФ = Inp_BBTimeframe): свеча1 закрылась ниже нижней     |
//| полосы BB, свеча2 закрылась выше свечи1, тренд H4 = BULL. Вход  |
//| на открытии следующей свечи. Признаки - M5, 20 лагов (lag0 свежий).|
//| OnTick тикает по M1, поэтому если Inp_BBTimeframe > M1, один и  |
//| тот же подтверждённый бар сигнала может "стоять" несколько      |
//| M1-баров подряд - g_last_bb_bar защищает от повторного входа.  |
//+------------------------------------------------------------------+
void DetectAndTrade()
  {
   datetime signal_time = iTime(_Symbol, Inp_BBTimeframe, 1);   // свеча подтверждения (ТФ сигнала)
   if(signal_time == g_last_bb_bar)
      return;   // этот бар Inp_BBTimeframe уже обработан на предыдущих M1-тиках

   // --- Свечи ТФ сигнала: закрытия и нижняя полоса Боллинджера (3 бара) ---
   double close1[], bb_lower1[];
   if(CopyClose(_Symbol, Inp_BBTimeframe, 0, 3, close1) < 3)
      return;
   if(CopyBuffer(h_bb_sig, LOWER_BAND, 0, 3, bb_lower1) < 3)
      return;

   g_last_bb_bar = signal_time;   // помечаем бар обработанным (независимо от результата ниже)

   // Свеча 1 (shift 2): закрылась НИЖЕ нижней полосы
   if(close1[2] >= bb_lower1[2])
      return;
   // Свеча 2 (shift 1): закрылась ВЫШЕ свечи 1 (подтверждение разворота)
   if(close1[1] <= close1[2])
      return;
   // Тренд H4 = BULL
   if(!H4_TrendFilter_IsBullish())
      return;

   if(IsLastBarOfDay(signal_time, Inp_BBTimeframe))
      return;   // сигналы на последнем баре дня не открываем

   buy_signals++;

   // --- Признаки на M5: лаг0 = бар M5 на момент сигнала ---
   int m5_shift = iBarShift(_Symbol, FEATURE_PERIOD, signal_time, false);
   if(m5_shift < 0)
      return;
   if(m5_shift + EXPORT_SEQ_LEN > iBars(_Symbol, FEATURE_PERIOD))
      return;   // не хватает истории для окна

   double open5[], close5[], high5[], low5[];
   double ema8_5[], ema21_5[], rsi_5[], atr_5[];
   double bb_upper5[], bb_mid5[], bb_lower5[];

   if(CopyOpen(_Symbol, FEATURE_PERIOD, m5_shift, EXPORT_SEQ_LEN, open5) < EXPORT_SEQ_LEN) return;
   if(CopyClose(_Symbol, FEATURE_PERIOD, m5_shift, EXPORT_SEQ_LEN, close5) < EXPORT_SEQ_LEN) return;
   if(CopyHigh(_Symbol, FEATURE_PERIOD, m5_shift, EXPORT_SEQ_LEN, high5) < EXPORT_SEQ_LEN) return;
   if(CopyLow(_Symbol, FEATURE_PERIOD, m5_shift, EXPORT_SEQ_LEN, low5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_ema8, 0, m5_shift, EXPORT_SEQ_LEN, ema8_5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_ema21, 0, m5_shift, EXPORT_SEQ_LEN, ema21_5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_rsi, 0, m5_shift, EXPORT_SEQ_LEN, rsi_5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_atr, 0, m5_shift, EXPORT_SEQ_LEN, atr_5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_bb, UPPER_BAND, m5_shift, EXPORT_SEQ_LEN, bb_upper5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_bb, BASE_LINE, m5_shift, EXPORT_SEQ_LEN, bb_mid5) < EXPORT_SEQ_LEN) return;
   if(CopyBuffer(h_bb, LOWER_BAND, m5_shift, EXPORT_SEQ_LEN, bb_lower5) < EXPORT_SEQ_LEN) return;

   double features[];
   ArrayResize(features, total_features);
   BuildFeatureVector(open5, close5, high5, low5,
                      ema8_5, ema21_5, rsi_5, atr_5,
                      bb_upper5, bb_mid5, bb_lower5,
                      EXPORT_SEQ_LEN, features);

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl  = (Inp_SL_Points > 0.0) ? ask - sl_price : 0.0;
   double tp  = (Inp_TP_Points > 0.0) ? ask + tp_price : 0.0;

   if(trade.Buy(Inp_LotSize, _Symbol, ask, sl, tp, "RNN_Data"))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, signal_time, features);
         buy_trades++;
        }
     }
  }
//+------------------------------------------------------------------+
//| Добавление сделки в список отслеживания                          |
//+------------------------------------------------------------------+
void AddDeal(ulong ticket, datetime signal_time, double &features[])
  {
   int idx = deals_count++;
   ArrayResize(deals, deals_count);
   deals[idx].ticket      = ticket;
   deals[idx].signal_time = signal_time;
   deals[idx].held_bars   = 0;
   ArrayResize(deals[idx].features, total_features);
   ArrayCopy(deals[idx].features, features);
  }
//+------------------------------------------------------------------+
//| Удаление сделки из списка отслеживания                           |
//+------------------------------------------------------------------+
void RemoveDeal(int idx)
  {
   deals_count--;
   if(idx < deals_count)
     {
      deals[idx].ticket      = deals[deals_count].ticket;
      deals[idx].signal_time = deals[deals_count].signal_time;
      deals[idx].held_bars   = deals[deals_count].held_bars;
      ArrayCopy(deals[idx].features, deals[deals_count].features);
     }
   ArrayResize(deals[deals_count].features, 0);
   ArrayResize(deals, deals_count);
  }
//+------------------------------------------------------------------+
//| Запись строки в файл                                             |
//+------------------------------------------------------------------+
//| Отчёт-статистика по разметке: пишется рядом с CSV (Common/Files) |
//| Имя: <имя CSV без .csv>_stats.txt                                |
//+------------------------------------------------------------------+
void WriteStatsReport(const string csv_name)
  {
   string base = csv_name;
   StringReplace(base, ".csv", "");
   if(base == "")
      base = "data_BUY";
   string fname = base + "_stats.txt";

   if(FileIsExist(fname, FILE_COMMON))
      FileDelete(fname, FILE_COMMON);

   CFileTxt rep;
   if(!rep.Open(fname, FILE_TXT|FILE_WRITE|FILE_ANSI|FILE_COMMON))
     {
      PrintFormat("PrepareData: не удалось создать отчёт статистики '%s' (код %d)", fname, GetLastError());
      return;
     }

   int    total  = buy_label1 + buy_label0;
   double p_l1   = (total > 0)      ? 100.0 * buy_label1 / (double)total         : 0.0;
   double p_l0   = (total > 0)      ? 100.0 * buy_label0 / (double)total         : 0.0;
   double p_tp   = (buy_label1 > 0) ? 100.0 * g_stat_tp  / (double)buy_label1    : 0.0;
   double p_thr  = (buy_label1 > 0) ? 100.0 * g_stat_thr / (double)buy_label1    : 0.0;
   double p_sl   = (buy_label0 > 0) ? 100.0 * g_stat_sl  / (double)buy_label0    : 0.0;
   double p_oth  = (buy_label0 > 0) ? 100.0 * g_stat_other / (double)buy_label0  : 0.0;

   rep.WriteString("================================================================\n");
   rep.WriteString("PrepareData v2.03 - ОТЧЁТ О РАЗМЕТКЕ ДАТАСЕТА\n");
   rep.WriteString("================================================================\n");
   rep.WriteString(StringFormat("CSV файл:               %s\n", csv_name));
   rep.WriteString(StringFormat("Символ:                 %s\n", _Symbol));
   rep.WriteString(StringFormat("Интервал теста:         %s - %s\n",
                               TimeToString(test_start, TIME_DATE|TIME_MINUTES),
                               TimeToString(test_end,   TIME_DATE|TIME_MINUTES)));
   rep.WriteString(StringFormat("Таймфрейм сигнала:      %s\n", EnumToString(Inp_BBTimeframe)));
   rep.WriteString(StringFormat("Сигнал:                 BB(%d, %.2f) на %s: закрытие ниже нижней полосы + подтверждение закрытием выше + тренд H4=BULL\n",
                               Inp_BB_Period, Inp_BB_Dev, EnumToString(Inp_BBTimeframe)));
   rep.WriteString(StringFormat("TP / SL (пункты):       %.1f / %.1f\n", Inp_TP_Points, Inp_SL_Points));
   rep.WriteString(StringFormat("Горизонт удержания:     %d баров %s\n", Inp_HoldBars, EnumToString(SIGNAL_PERIOD)));
   rep.WriteString(StringFormat("Порог прибыли (у.е.):   %.2f %s\n", Inp_ProfitThresholdUSD,
                               (Inp_ProfitThresholdUSD > 0.0) ? "(метка [1,0] = TP ИЛИ результат >= порога)"
                                                              : "(выключен: [1,0] только по TP)"));
   rep.WriteString(StringFormat("Лот / магик:            %.2f / %d\n", Inp_LotSize, Inp_Magic));
   rep.WriteString(StringFormat("Признаки:               M5, %d баров x %d признаков\n", EXPORT_SEQ_LEN, EXPORT_FEATURES));
   rep.WriteString("----------------------------------------------------------------\n");
   rep.WriteString("МЕТКИ\n");
   rep.WriteString("----------------------------------------------------------------\n");
   rep.WriteString(StringFormat("Всего меток:             %d\n", total));
   rep.WriteString(StringFormat("[1,0] (положительные):   %d (%.2f%%)\n", buy_label1, p_l1));
   rep.WriteString(StringFormat("[0,1] (остальные):       %d (%.2f%%)\n", buy_label0, p_l0));
   rep.WriteString(StringFormat("РАЗБИВКА [1,0] (всего %d):\n", buy_label1));
   rep.WriteString(StringFormat("  закрыто строго по TP:            %d (%.2f%% от [1,0])\n", g_stat_tp, p_tp));
   rep.WriteString(StringFormat("  выше порога (закрыто не по TP):  %d (%.2f%% от [1,0])\n", g_stat_thr, p_thr));
   rep.WriteString(StringFormat("РАЗБИВКА [0,1] (всего %d):\n", buy_label0));
   rep.WriteString(StringFormat("  закрыто по SL:                   %d (%.2f%% от [0,1])\n", g_stat_sl, p_sl));
   rep.WriteString(StringFormat("  прочее (горизонт/день/OnDeinit): %d (%.2f%% от [0,1])\n", g_stat_other, p_oth));
   rep.WriteString("----------------------------------------------------------------\n");
   rep.WriteString("СДЕЛКИ\n");
   rep.WriteString("----------------------------------------------------------------\n");
   rep.WriteString(StringFormat("Сигналов обнаружено:     %d\n", buy_signals));
   rep.WriteString(StringFormat("Сделок открыто:          %d\n", buy_trades));
   rep.WriteString(StringFormat("Строк записано:          %d\n", rows_written));
   rep.WriteString("================================================================\n");
   rep.Close();

   PrintFormat("PrepareData: отчёт статистики сохранён: Common/Files/%s", fname);
  }
//+------------------------------------------------------------------+
//| Запись строки в файл (метка + статистика)                        |
//+------------------------------------------------------------------+
void WriteDealRow(datetime time, double &features[], double result, ENUM_EXIT_KIND exit_kind)
  {
   bool by_tp     = (exit_kind == EXIT_TP);
   bool above_thr = (!by_tp && Inp_ProfitThresholdUSD > 0.0 && result >= Inp_ProfitThresholdUSD);
   string label   = (by_tp || above_thr) ? "[1,0]" : "[0,1]";

   if(Inp_ExportData)
      WriteRowToCSV(csvFileBuy, time, "BUY", features, result, label);
   if(label == "[1,0]")
     {
      buy_label1++;
      if(by_tp) g_stat_tp++;
      else      g_stat_thr++;
     }
   else
     {
      buy_label0++;
      if(exit_kind == EXIT_SL) g_stat_sl++;
      else                     g_stat_other++;
     }
  }
//+------------------------------------------------------------------+
//| Запись заголовка в CSV: datetime,signal,lag0_*,...,lag19_*,     |
//|   lag0 - сигнальный бар (самый свежий), порядок колонок         |
//|   совпадает с порядком признаков в векторе.                     |
//+------------------------------------------------------------------+
void WriteHeader(CFileTxt &file)
  {
   file.WriteString("datetime,signal");
   string names[EXPORT_FEATURES] = {"open","high","low","close","direction",
                                    "ema8","ema21","rsi","atr",
                                    "bb_upper","bb_mid","bb_lower"};
   for(int lag = 0; lag < EXPORT_SEQ_LEN; lag++)
      for(int i = 0; i < EXPORT_FEATURES; i++)
         file.WriteString("," + "lag" + IntegerToString(lag) + "_" + names[i]);
   file.WriteString(",result,label\n");
  }
//+------------------------------------------------------------------+
//| Заполнение OHLC, индикаторов и дополнительных признаков бара.   |
//+------------------------------------------------------------------+
void FillBarFeatures(int bar,
                     const double &open[],
                     const double &close[],
                     const double &high[],
                     const double &low[],
                     const double &ema8[],
                     const double &ema21[],
                     const double &rsi[],
                     const double &atr[],
                     const double &bb_upper[],
                     const double &bb_mid[],
                     const double &bb_lower[],
                     double &out[])
  {
   out[0]  = open[bar];
   out[1]  = high[bar];
   out[2]  = low[bar];
   out[3]  = close[bar];
   out[4]  = (close[bar] > open[bar]) ? 1.0 : -1.0;   // направление свечи
   out[5]  = ema8[bar];
   out[6]  = ema21[bar];
   out[7]  = rsi[bar];
   out[8]  = atr[bar];
   out[9]  = bb_upper[bar];
   out[10] = bb_mid[bar];
   out[11] = bb_lower[bar];
  }
//+------------------------------------------------------------------+
//| Формирование вектора признаков (seq_len x 12).                  |
//| Внимание: f0..f11 - сигнальный бар (самый свежий в окне),       |
//| далее - более старые бары (порядок: новое -> старое).           |
//+------------------------------------------------------------------+
void BuildFeatureVector(const double &open[],
                        const double &close[],
                        const double &high[],
                        const double &low[],
                        const double &ema8[],
                        const double &ema21[],
                        const double &rsi[],
                        const double &atr[],
                        const double &bb_upper[],
                        const double &bb_mid[],
                        const double &bb_lower[],
                        int seq_len,
                        double &features[])
  {
   double bar_feat[EXPORT_FEATURES];
   int idx = 0;
   for(int k = 0; k < seq_len; k++)
     {
      FillBarFeatures(k, open, close, high, low,
                      ema8, ema21, rsi, atr,
                      bb_upper, bb_mid, bb_lower, bar_feat);
      for(int f = 0; f < EXPORT_FEATURES; f++)
         features[idx++] = bar_feat[f];
     }
  }
//+------------------------------------------------------------------+
//| Запись строки в CSV                                              |
//+------------------------------------------------------------------+
void WriteRowToCSV(CFileTxt &file, datetime time, string signal, double &features[], double result, string label)
  {
   string row = StringFormat("%s,%s", TimeToString(time), signal);
   int size = ArraySize(features);
   for(int i = 0; i < size; i++)
      row += StringFormat(",%.6f", features[i]);
   // label берём в кавычки, чтобы запись [1,0] не разбилась на две колонки
   row += StringFormat(",%.2f,\"%s\"\n", result, label);
   file.WriteString(row);
   rows_written++;
  }
//+------------------------------------------------------------------+
