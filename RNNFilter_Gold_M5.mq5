//+------------------------------------------------------------------+
//|                                              RNNFilter_Gold_M5.mq5 |
//|                          Copyright 2026, Bondarev A.            |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Bondarev A."
#property link      "https://www.mql5.com"
#property version   "1.07"
//+------------------------------------------------------------------+
//|                    RNN Filter Expert EA                          |
//|   Торговля через фильтрацию сигналов GRU-нейросетью (ONNX)       |
//|   BUY-only: торгуем ТОЛЬКО в лонг, сделок по продаже нет.        |
//|   ВСЕГДА только ОДНА активная сделка.                            |
//|   BUY проходит, если BUY-сеть уверена в покупке.                 |
//|   Фильтр тренда H1 (EMA50 + ADX + ATR) отсекает контртренд.      |
//|   В строку состояния (Comment) выводится confidence BUY-сети.    |
//|   Серии и множители лота удалены — фиксированный лот.            |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include "include/RNN_ONNX.mqh"
#include "include/H4_trend_filter.mqh"

#define SIGNAL_PERIOD    PERIOD_M1
#define FEATURE_PERIOD   PERIOD_M5
#define TREND_PERIOD     PERIOD_H4

//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
// --- Включение ---
input group "Основные настройки"
input bool   Inp_Enabled          = true;      // Включить советник
input bool   Inp_BuySignal        = true;      // Учитывать BUY сигналы
input bool   Inp_UseNNFilter      = true;      // Использовать фильтр НС (ONNX)
input bool   Inp_TradeWOIndicators = false;    // Торговать БЕЗ индикаторных сигналов (только по сетям)

// --- Порог уверенности BUY-сети ---
input group "Нейросеть (ONNX)"
input double Inp_BuyConfThreshold  = 0.7;      // BUY-сеть: уверена в покупке, если conf >= порог
input int    Inp_NNProfitClass     = 1;        // Класс "прибыльной" сделки (0/1)

// --- Путь к ONNX-модели BUY (относительно Common\Files или MQL5\Files) ---

input string Inp_BuyModelFile     = "RNN_GOLD_M15\\buy_model.onnx";   // Файл модели BUY

// --- Торговые настройки (фиксированный лот, без серий/множителей) ---
input group "Торговые настройки"
input double Inp_LotSize          = 0.01;      // Объём лота (фиксированный)
input int    Inp_Magic            = 777111;    // Магик-номер

input double Inp_SL_Points        = 150.0;     // Стоп-лосс (пункты), 0 = без SL
input double Inp_TP_Points        = 200.0;     // Тейк-профит (пункты), 0 = без TP

// --- Bollinger Bands (сигнальная детекция) ---
input group "Сигнал (Bollinger)"
input ENUM_TIMEFRAMES Inp_BBTimeframe = PERIOD_M1;  // Таймфрейм сигнала: BB + закрытие свечей
input int    Inp_BB_Period   = 20;   // Боллинджер: период
input double Inp_BB_Dev      = 2.0;  // Боллинджер: множитель (сигма)

// --- Трейлинг ---
input group "Трейлинг"
input bool   Inp_TrailingEnable   = false;     // Включить трейлинг
input double Inp_TrailingPoints   = 100.0;     // Трейлинг-дистанция (пункты)
input double Inp_TrailingStartPoints = 50.0;   // Стартовый профит для трейлинга (пункты)
input bool   Inp_TrailingLogEnable  = true;    // Логировать трейлинг

// --- Безубыток ---
input group "Безубыток"
input bool   Inp_BreakEvenEnable  = false;     // Включить безубыток
input double Inp_BreakEvenPoints  = 50.0;      // Безубыток уровень (пункты)
input double Inp_BEOffsetPoints   = 20.0;      // Смещение от цены (пункты)
input bool   Inp_BreakEvenLogEnable = true;    // Логировать безубыток

// --- Удержание ---
input group "Удержание"
input int    Inp_HoldBars         = 20;        // Макс. горизонт удержания (бары)

// --- Фильтр спреда ---
input group "Фильтр спреда"
input bool   Inp_SpreadEnable     = false;     // Включить фильтр спреда
input int    Inp_MaxSpreadPoints  = 30;        // Макс. спред (пункты)

// --- Фильтр тренда H4 (главный фильтр направления) ---
input group "Фильтр тренда H4"
input int    Inp_ADX_Period    = 14;    // ADX/+DI/-DI: период
input double Inp_ADX_Threshold = 25.0;  // ADX: порог тренда (классика)

// --- Пользовательский критерий оптимизации (winrate) ---
#include "include/TargetWinRateCriterion.mqh"
//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CTrade                  trade;

// Индикаторные хендлы: 0=EMA8, 1=EMA21, 2=RSI, 3=ATR (признаки сети)
int                     handles[4];
int                     handle_bb = INVALID_HANDLE;         // Bollinger Bands M5 (признаки сети)
int                     h_bb_sig = INVALID_HANDLE;          // Bollinger Bands на Inp_BBTimeframe (сигнал)

// Структура записи сделки (всегда не более одной активной)
struct DealRecord
  {
   ulong         ticket;
   int           type;              // 0=BUY (BUY-only)
   datetime      signal_time;
   int           held_bars;
   double        entry_price;
   double        sl;
   double        tp;
  };

DealRecord            deals[];
int                   deals_count = 0;

// Последний confidence BUY-сети (для строки состояния)
double                g_last_conf_buy  = 0.0;

// Последние значения фильтра тренда H1 (для отображения на графике)
double                g_last_adx_h1    = 0.0;   // ADX H1
double                g_last_ratio_h1  = 0.0;   // |close-EMA|/ATR H1
int                   g_last_trend_dir = 0;     // H1: 1=вверх, -1=вниз, 0=флэт/нет данных

string                g_last_decision  = "PASS";
string                g_last_reason    = "";         // причина последнего решения (для лога)
// --- Сигнал BB (дедуп-гвард)
datetime              g_last_bb_bar = 0;            // дедуп-гвард BB-сигнала
double                tp_price = 0.0;               // TP в пунктах (новое)


// --- Графические панели (OBJ_LABEL): слева - мониторинг, справа - инфо о сделке ---
string                g_panel_left  = "RNNFilter_Monitor";
string                g_panel_right = "RNNFilter_Trade";
string                g_trade_info  = "Сделка: не открывалась";

// --- Перемещаемые панели (по статье mql5.com/ru/articles/12923) ---
// Торговая панель (кнопка Buy + лот) и панель логов
string                g_pan = "RNN_Panel";      // префикс торговой панели
string                g_logp= "RNN_Log";        // префикс панели логов
#define PANEL_OFFSET_X   190                    // горизонтальный сдвиг лог-панели
#define PANEL_OFFSET_Y   20
#define DRAG_TOLERANCE   3

// Координаты текущей позиции перемещения панелей
int                   g_panel_drag_x = 0;        // X заголовка при захвате (изменяемая)
int                   g_panel_drag_y = 0;
int                   g_panel_mid_x  = 0;
int                   g_panel_mid_y  = 0;

// Лог-буфер
string                g_log_lines[];

// --- Состояние перетаскивания панелей (мышь) ---
bool                  g_drag_pan  = true;
bool                  g_drag_log  = true;
int                   g_drag_dx   = 0;
int                   g_drag_dy   = 0;

int                   nn_buy_seq_len = 0;
datetime              last_bar_time = 0;

double                sl_price = 0.0;
double                trail_points = 0.0;
double                trail_start = 0.0;
double                be_points = 0.0;
double                be_offset = 0.0;
string                g_price_fmt = "%.2f";

// Буферы для индикаторов и рыночных данных
double                open_buf[], close_buf[], high_buf[], low_buf[];

double                ema8_buf[], ema21_buf[], rsi_buf[];
double                atr_buf[];
double                bb_upper_buf[], bb_mid_buf[], bb_lower_buf[];   // Bollinger (признаки сети)
//+------------------------------------------------------------------+
//| Инициализация                                                    |
//+------------------------------------------------------------------+
int OnInit()
  {
   // --- Валидация параметров ---
   if(Inp_HoldBars < 1 || Inp_LotSize <= 0)
     {
      Print("Ошибка параметров: HoldBars>=1, LotSize>0");
      return(INIT_FAILED);
     }

   g_price_fmt = StringFormat("%%.%df", _Digits);

   // --- Валидация порога НС ---
   if(Inp_BuyConfThreshold <= 0.0)
     {
      Print("Ошибка порога НС: порог уверенности должен быть > 0");
      return(INIT_FAILED);
     }

   nn_buy_seq_len = NN_ONNX_GetSeqLen(0);
   if(nn_buy_seq_len < 1 || nn_buy_seq_len > 60)
     {
      PrintFormat("Ошибка: некорректное окно входа НС BUY=%d (ожидалось 1..60)", nn_buy_seq_len);
      return(INIT_FAILED);
     }

   sl_price = Inp_SL_Points * _Point;
   tp_price = Inp_TP_Points * _Point;
   trail_points = Inp_TrailingPoints * _Point;
   trail_start = Inp_TrailingStartPoints * _Point;
   be_points = Inp_BreakEvenPoints * _Point;
   be_offset = Inp_BEOffsetPoints * _Point;

   // --- Создаём индикаторные хендлы ---
   handles[0] = iMA(_Symbol, PERIOD_M5, 8,  0, MODE_EMA, PRICE_CLOSE);
   handles[1] = iMA(_Symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
   handles[2] = iRSI(_Symbol, PERIOD_M5, 14, PRICE_CLOSE);
   handles[3] = iATR(_Symbol, PERIOD_M5, 14);
   handle_bb  = iBands(_Symbol, FEATURE_PERIOD, Inp_BB_Period, 0, Inp_BB_Dev, PRICE_CLOSE);
   h_bb_sig = iBands(_Symbol, Inp_BBTimeframe, Inp_BB_Period, 0, Inp_BB_Dev, PRICE_CLOSE);   // Bollinger для сети

   for(int i = 0; i < 4; i++)
     {
      if(handles[i] == INVALID_HANDLE)
        {
         PrintFormat("Ошибка создания индикатора %d (код: %d)", i, GetLastError());
         return(INIT_FAILED);
        }
     }

   // --- Проверка BB-хендла ---
   if(handle_bb == INVALID_HANDLE)
     {
      PrintFormat("Ошибка создания индикатора Bollinger (код: %d)", GetLastError());
      return(INIT_FAILED);
     }

   // --- Инициализация фильтра тренда H4 ---
   if(!H4_TrendFilter_Init(Inp_ADX_Period, Inp_ADX_Threshold))
      return(INIT_FAILED);

   // --- Настраиваем торговый объект ---
   trade.SetExpertMagicNumber(Inp_Magic);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFillingBySymbol(_Symbol);

   // --- Инициализируем ONNX-модели (только BUY) ---
   if(Inp_UseNNFilter)
     {
      if(!NN_ONNX_Init(Inp_BuyModelFile, ""))
        {
         Print("RNNFilter: не удалось загрузить ONNX-модель BUY. Файл должен лежать в Common\\Files\\RNN_GOLD\\");
         return(INIT_FAILED);
        }
     }

   PrintFormat("RNNFilter инициализирован. NN: ONNX (%s), порог BUY=%.2f, SL: %.1f п., TP: %.1f п., Hold: %d баров M1",
               Inp_BuyModelFile,
               Inp_BuyConfThreshold,
               Inp_SL_Points, Inp_TP_Points, Inp_HoldBars);

   UpdateStatus();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Деинициализация                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Закрываем оставшиеся позиции
   CloseAllDeals();

   // Освобождаем ONNX-сессии
   if(Inp_UseNNFilter)
      NN_ONNX_Deinit();

   // Освобождаем индикаторные хендлы
   for(int i = 0; i < 4; i++)
     {
      if(handles[i] != INVALID_HANDLE)
         IndicatorRelease(handles[i]);
     }

   if(handle_bb != INVALID_HANDLE)
      IndicatorRelease(handle_bb);

   if(h_bb_sig != INVALID_HANDLE)
      IndicatorRelease(h_bb_sig);

   // Деинициализация фильтра тренда H4
   H4_TrendFilter_Deinit();

   PrintFormat("RNNFilter деинициализирован. Сделок обработано: %d", deals_count);
  }
//+------------------------------------------------------------------+
//| Обработка каждого тика                                           |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Глобальный выключатель
   if(!Inp_Enabled)
     {
      UpdateStatus();
      return;
     }

   datetime cur_bar = iTime(_Symbol, SIGNAL_PERIOD, 0);
   if(cur_bar == last_bar_time)
     {
      UpdateStatus();
      return;  // Та же самая свеча — ничего не делаем
     }

   last_bar_time = cur_bar;

   // На последнем баре дня закрываем все позиции
   if(IsLastBarOfDay(cur_bar, SIGNAL_PERIOD))
     {
      CloseAllDeals();
      UpdateStatus();
      return;  // новые сигналы в конце дня не открываем
     }

   // --- Обрабатываем открытые сделки ---
   ProcessOpenDeals();

   // --- Трейлинг и безубыток ---
   if(Inp_TrailingEnable)
      TrailingPositions();
   if(Inp_BreakEvenEnable)
      MoveToBreakEven();

   // --- Фильтр спреда ---
   if(Inp_SpreadEnable && !IsSpreadOK())
     {
      UpdateStatus();
      return;
     }

   // --- Обновляем тренд H4 ---
   H4_TrendFilter_Update(Inp_ADX_Threshold);

   // --- Проверяем сигналы ---
   CheckSignals();

   // --- Строка состояния ---
   UpdateStatus();
  }
//+------------------------------------------------------------------+
//| Признак последнего бара торгового дня                            |
//+------------------------------------------------------------------+
bool IsLastBarOfDay(datetime bar_time, ENUM_TIMEFRAMES period = SIGNAL_PERIOD)
  {
   MqlDateTime dt, dt_next;
   TimeToStruct(bar_time, dt);
   TimeToStruct(bar_time + PeriodSeconds(period), dt_next);
   return(dt.day != dt_next.day);
  }
//+------------------------------------------------------------------+
//| Есть ли уже открытая сделка (только одна активная)               |
//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != Inp_Magic)
         continue;
      return(true);
     }
   return(false);
  }
//+------------------------------------------------------------------+
//| Принудительное закрытие всех позиций до конца дня                |
//+------------------------------------------------------------------+
void CloseAllDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         RemoveDeal(i);
         continue;
        }

      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      if(trade.PositionClose(deals[i].ticket))
        {
         PrintFormat("Закрыта сделка #%I64u, профит: %.2f", deals[i].ticket, profit);
        }
      else
        {
         PrintFormat("Ошибка закрытия сделки #%I64u", deals[i].ticket);
        }
      RemoveDeal(i);
     }
  }
//+------------------------------------------------------------------+
//| Обработка открытых сделок                                        |
//+------------------------------------------------------------------+
void ProcessOpenDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      // Проверяем, закрыта ли позиция сервером
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         RemoveDeal(i);
         continue;
        }

      deals[i].held_bars++;

      // Проверяем горизонт удержания
      if(deals[i].held_bars >= Inp_HoldBars)
        {
         if(trade.PositionClose(deals[i].ticket))
           {
            PrintFormat("Досрочное закрытие #%I64u по горизонту (%d баров)",
                        deals[i].ticket, deals[i].held_bars);
           }
         RemoveDeal(i);
         continue;
        }
     }
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
      deals[idx].type        = deals[deals_count].type;
      deals[idx].signal_time = deals[deals_count].signal_time;
      deals[idx].held_bars   = deals[deals_count].held_bars;
      deals[idx].entry_price = deals[deals_count].entry_price;
      deals[idx].sl          = deals[deals_count].sl;
      deals[idx].tp          = deals[deals_count].tp;
     }
   ArrayResize(deals, deals_count);
  }
//+------------------------------------------------------------------+
//| Трейлинг позиций (BUY-only)                                      |
//+------------------------------------------------------------------+
void TrailingPositions()
  {
   for(int i = 0; i < deals_count; i++)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
         continue;

      double pos_sl = PositionGetDouble(POSITION_SL);
      double entry  = PositionGetDouble(POSITION_PRICE_OPEN);

      // BUY-only: все сделки в лонг
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if((bid - entry) < trail_start)
         continue;

      double new_sl = bid - trail_points;
      double profit_pts = (bid - entry) / _Point;
      if(Inp_TrailingLogEnable)
         PrintFormat("[TR-LOG] %s | #%I64u | профит %.1f п. | дистанция %.1f п. | newSL %.2f (текущий SL %.2f)",
                     TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES),
                     deals[i].ticket, profit_pts, trail_points / _Point,
                     new_sl, pos_sl);
      if(new_sl > pos_sl && new_sl > entry)
        {
         double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
         if(trade.PositionModify(deals[i].ticket, new_sl, tp))
            {
            deals[i].sl = new_sl;
            if(Inp_TrailingLogEnable)
               PrintFormat("[TR-LOG] %s | #%I64u | OK: SL->%.2f",
                           TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES),
                           deals[i].ticket, new_sl);
            }
        }
     }
  }
//+------------------------------------------------------------------+
//| Перевод в безубыток (BUY-only)                                   |
//+------------------------------------------------------------------+
void MoveToBreakEven()
  {
   for(int i = 0; i < deals_count; i++)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
         continue;

      double pos_sl = PositionGetDouble(POSITION_SL);
      double entry  = PositionGetDouble(POSITION_PRICE_OPEN);

      // BUY-only
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if((bid - entry) < be_points)
         continue;

      double new_sl = entry + be_offset;
      double new_sl_pts = (new_sl - entry) / _Point;
      double profit_pts  = (bid - entry) / _Point;
      if(Inp_BreakEvenLogEnable)
         PrintFormat("[BE-LOG] %s | #%I64u | активация: профит %.1f п. (порог %.1f п.) | перенос SL: %.2f -> %.2f (смещение +%.1f п.)",
                  TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES),
                  deals[i].ticket, profit_pts, be_points / _Point,
                  pos_sl, new_sl, new_sl_pts);
      if(pos_sl < new_sl)
        {
         double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
         if(trade.PositionModify(deals[i].ticket, new_sl, tp))
            {
            deals[i].sl = new_sl;
            if(Inp_BreakEvenLogEnable)
               PrintFormat("[BE-LOG] %s | #%I64u | успешно: SL->%.2f",
                        TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES),
                        deals[i].ticket, new_sl);
            }
        }
     }
  }
//+------------------------------------------------------------------+
//| Проверка спреда                                                  |
//+------------------------------------------------------------------+
bool IsSpreadOK()
  {
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return(spread <= Inp_MaxSpreadPoints);
  }
//+------------------------------------------------------------------+
//| Нормализация объёма лота                                         |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
  {
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lot_step <= 0.0)
      lot_step = 0.01;

   lot = MathMax(MathMin(lot, max_lot), min_lot);
   lot = MathRound(lot / lot_step) * lot_step;
   return(NormalizeDouble(lot, 8));
  }
//+------------------------------------------------------------------+
//| Проверка сигналов и открытие сделок                              |
//+------------------------------------------------------------------+
void LogSignal(datetime signal_time, bool buy_signal, double conf_buy, bool nn_active)
  {
   string net = "NN off";
   if(nn_active)
      net = (conf_buy >= Inp_BuyConfThreshold) ? "УВЕРЕНА (BUY)" : "не уверена";
   PrintFormat("[RNN_LOG] %s | BUY-сигнал: %s | BUYconf=%.4f (порог %.2f) | Сеть: %s | Решение: %s | %s",
               TimeToString(signal_time, TIME_DATE | TIME_MINUTES),
               buy_signal ? "ДА" : "нет",
               conf_buy, Inp_BuyConfThreshold,
               net, g_last_decision, g_last_reason);
  }
//+------------------------------------------------------------------+
//| Проверка сигналов и открытие сделок                              |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Детекция BB-сигнала (разворот у нижней полосы)                  |
//+------------------------------------------------------------------+
bool DetectBBSignal(datetime &signal_time)
  {
   signal_time = iTime(_Symbol, Inp_BBTimeframe, 1);
   if(signal_time == g_last_bb_bar)
      return false;

   double close1[], bb_lower1[];
   if(CopyClose(_Symbol, Inp_BBTimeframe, 0, 3, close1) < 3)
      return false;
   if(CopyBuffer(h_bb_sig, LOWER_BAND, 0, 3, bb_lower1) < 3)
      return false;

   g_last_bb_bar = signal_time;

   if(close1[2] >= bb_lower1[2])
      return false;
   if(close1[1] <= close1[2])
      return false;
   if(!H4_TrendFilter_IsBullish())
      return false;
   if(IsLastBarOfDay(signal_time, Inp_BBTimeframe))
      return false;

   return true;
  }

void CheckSignals()
  {
   // Детектируем BB-сигнал
   datetime signal_time;
   bool buy_signal = DetectBBSignal(signal_time);

   // Ни сигнала, ни режима БЕЗ индикаторов — ничего не делаем
   if(!buy_signal && !Inp_TradeWOIndicators)
     {
      UpdateStatus();
      return;
     }

   // --- Сбор признаков M5 (если есть сигнал или режим без индикаторов) ---
   int m5_shift = iBarShift(_Symbol, FEATURE_PERIOD, signal_time, false);
   if(m5_shift < 0)
     {
      UpdateStatus();
      return;
     }
   if(m5_shift + NN_BUY_NORM_WINDOW > iBars(_Symbol, FEATURE_PERIOD))
     {
      UpdateStatus();
      return;
     }

   double open5[], close5[], high5[], low5[];
   double ema8_5[], ema21_5[], rsi_5[], atr_5[];
   double bb_upper5[], bb_mid5[], bb_lower5[];

   if(CopyOpen(_Symbol, FEATURE_PERIOD, m5_shift, NN_BUY_NORM_WINDOW, open5) < NN_BUY_NORM_WINDOW ||
      CopyClose(_Symbol, FEATURE_PERIOD, m5_shift, NN_BUY_NORM_WINDOW, close5) < NN_BUY_NORM_WINDOW ||
      CopyHigh(_Symbol, FEATURE_PERIOD, m5_shift, NN_BUY_NORM_WINDOW, high5) < NN_BUY_NORM_WINDOW ||
      CopyLow(_Symbol, FEATURE_PERIOD, m5_shift, NN_BUY_NORM_WINDOW, low5) < NN_BUY_NORM_WINDOW)
     {
      UpdateStatus();
      return;
     }

   if(CopyBuffer(handles[0], 0, m5_shift, NN_BUY_NORM_WINDOW, ema8_5) < NN_BUY_NORM_WINDOW ||
      CopyBuffer(handles[1], 0, m5_shift, NN_BUY_NORM_WINDOW, ema21_5) < NN_BUY_NORM_WINDOW ||
      CopyBuffer(handles[2], 0, m5_shift, NN_BUY_NORM_WINDOW, rsi_5) < NN_BUY_NORM_WINDOW ||
      CopyBuffer(handles[3], 0, m5_shift, NN_BUY_NORM_WINDOW, atr_5) < NN_BUY_NORM_WINDOW)
     {
      UpdateStatus();
      return;
     }

   if(CopyBuffer(handle_bb, UPPER_BAND, m5_shift, NN_BUY_NORM_WINDOW, bb_upper5) < NN_BUY_NORM_WINDOW ||
      CopyBuffer(handle_bb, BASE_LINE, m5_shift, NN_BUY_NORM_WINDOW, bb_mid5) < NN_BUY_NORM_WINDOW ||
      CopyBuffer(handle_bb, LOWER_BAND, m5_shift, NN_BUY_NORM_WINDOW, bb_lower5) < NN_BUY_NORM_WINDOW)
     {
      UpdateStatus();
      return;
     }

   double buy_features[];
   ArrayResize(buy_features, NN_BUY_NORM_WINDOW * NN_FEATURES);
   NN_BuildFeatures(0, open5, close5, high5, low5,
                    ema8_5, ema21_5, rsi_5, atr_5,
                    bb_upper5, bb_mid5, bb_lower5,
                    NN_BUY_NORM_WINDOW, NN_BUY_NORM_WINDOW, buy_features);

   // --- Прогон BUY-сети ONNX ---
   double conf_buy = 0.0;
   bool nn_active = (Inp_UseNNFilter && NN_ONNX_IsReady());

   if(nn_active)
     {
      conf_buy = NN_ONNX_Predict(0, buy_features, Inp_NNProfitClass);
      if(conf_buy < 0.0)
        {
         PrintFormat("RNNFilter: ошибка ONNX-инференса BUY (conf=%.2f)", conf_buy);
         UpdateStatus();
         return;
        }
     }

   g_last_conf_buy = conf_buy;

   // --- Решение: BUY (BUY-only) ---
   bool buy_ok = false;

   if(!Inp_TradeWOIndicators && buy_signal && Inp_BuySignal && !HasOpenPosition())
     {
      if(!nn_active)
         buy_ok = true;
      else if(conf_buy >= Inp_BuyConfThreshold)
         buy_ok = true;
     }

   // --- Режим БЕЗ индикаторных сигналов ---
   if(Inp_TradeWOIndicators && !HasOpenPosition() && nn_active)
     {
      if(Inp_BuySignal && conf_buy >= Inp_BuyConfThreshold)
         buy_ok = true;
     }

   // --- Причина решения ---
   if(buy_ok)
      g_last_reason = "сеть подтвердила (BUY)";
   else if(!buy_signal && !Inp_TradeWOIndicators)
      g_last_reason = "нет BB-сигнала";
   else if(HasOpenPosition())
      g_last_reason = "позиция уже открыта";
   else if(nn_active && conf_buy < Inp_BuyConfThreshold)
      g_last_reason = "сеть отклонила (BUYconf < порог)";
   else
      g_last_reason = "условия не выполнены";

   g_last_decision = (buy_ok ? "BUY" : "PASS");

   // --- ЛОГ ---
   PrintFormat("[RNN_LOG] %s | BB-сигнал: %s | BUYconf=%.4f (порог %.2f) | Сеть: %s | Решение: %s | %s",
               TimeToString(signal_time, TIME_DATE | TIME_MINUTES),
               buy_signal ? "ДА" : "нет",
               conf_buy, Inp_BuyConfThreshold,
               nn_active ? ((conf_buy >= Inp_BuyConfThreshold) ? "УВЕРЕНА" : "не уверена") : "off",
               g_last_decision, g_last_reason);

   // Открываем максимум одну сделку за бар
   if(buy_ok)
     {
      ProcessBuySignal(signal_time, conf_buy);
     }

   UpdateStatus();
  }

//+------------------------------------------------------------------+
//| Обработка BUY сигнала (проверка НС -> исполнение)                |
//+------------------------------------------------------------------+
void ProcessBuySignal(datetime signal_time, double conf_buy)
  {
   if(Inp_UseNNFilter)
     {
      // BUY разрешён, если BUY-сеть уверена в покупке
      if(conf_buy < Inp_BuyConfThreshold)
        {
         PrintFormat("BUY отклонён: BUYconf=%.4f (порог %.2f)",
                     conf_buy, Inp_BuyConfThreshold);
         return;
        }
     }

   ExecuteBuySignal(signal_time, conf_buy);
  }
//+------------------------------------------------------------------+
//| Исполнение BUY (только одна активная сделка, фикс. лот)          |
//+------------------------------------------------------------------+
void ExecuteBuySignal(datetime signal_time, double conf_buy)
  {
   // Только одна активная сделка
   if(HasOpenPosition())
      return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl  = (Inp_SL_Points > 0.0) ? ask - sl_price : 0.0;
   double tp  = (Inp_TP_Points > 0.0) ? ask + tp_price : 0.0;
   double lot = NormalizeLot(Inp_LotSize);

   string cmt = "RNN_Buy";
   if(Inp_UseNNFilter)
      cmt = StringFormat("B%.2f", conf_buy);

   if(trade.Buy(lot, _Symbol, ask, sl, tp, cmt))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, 0, signal_time, ask, sl, tp);
         g_trade_info = StringFormat("СДЕЛКА BUY  #%I64u\nЛот: %.2f\nЦена: " + g_price_fmt +
                                     "\nSL: " + g_price_fmt + "\nTP: " + g_price_fmt +
                                     "\nBUYconf = %.2f",
                                     ticket, lot, ask, sl, tp, conf_buy);
         PrintFormat("BUY #%I64u | Lot %.2f | Ask " + g_price_fmt + " | SL " + g_price_fmt + " | TP " + g_price_fmt + " | BUYconf=%.2f",
                     ticket, lot, ask, sl, tp, conf_buy);
        }
     }
  }
//+------------------------------------------------------------------+
//| Добавление сделки в список отслеживания                          |
//+------------------------------------------------------------------+
void AddDeal(ulong ticket, int type, datetime signal_time, double entry, double sl, double tp)
  {
   int idx = deals_count++;
   ArrayResize(deals, deals_count);
   deals[idx].ticket      = ticket;
   deals[idx].type        = type;
   deals[idx].signal_time = signal_time;
   deals[idx].held_bars   = 0;
   deals[idx].entry_price = entry;
   deals[idx].sl          = sl;
   deals[idx].tp          = tp;
  }
//+------------------------------------------------------------------+
//| Настройка/обновление текстовой метки-панели                      |
//+------------------------------------------------------------------+
void Panel_Render(const string name, const int corner, const int xd, const int yd,
                   const string text, const color txtclr, const color bgclr)
  {
   // --- Серый фон-подложка ---
   string bg = name + "_BG";
   if(ObjectFind(0, bg) < 0)
     {
      if(!ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0))
         return;
      ObjectSetInteger(0, bg, OBJPROP_CORNER, corner);
      ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, xd);
      ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, yd);
      ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, bg, OBJPROP_ZORDER, 0);
      ObjectSetInteger(0, bg, OBJPROP_BACK, true);   // на задний план, чтобы не перекрывать текст
      ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, bgclr);
      ObjectSetInteger(0, bg, OBJPROP_FILL, true);
      ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg, OBJPROP_WIDTH, 1);
     }

   // --- Размер фона по тексту ---
   string rows[];
   int n = StringSplit(text, '\n', rows);
   if(n < 1) n = 1;
   int maxlen = 0;
   for(int i = 0; i < n; i++)
      maxlen = MathMax(maxlen, StringLen(rows[i]));
   int fs = 10;
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, maxlen * (fs / 2) + 14);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, n * (fs + 2) + 8);

   // --- Текст поверх фона ---
   if(ObjectFind(0, name) < 0)
     {
      if(!ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0))
         return;
      ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xd + 7);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yd + 4);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_ZORDER, 1);
      ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fs);
     }
   ObjectSetInteger(0, name, OBJPROP_COLOR, txtclr);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
  }
//+------------------------------------------------------------------+
//| Удаление метки-панели (текст + фон)                              |
//+------------------------------------------------------------------+
void Panel_Delete(const string &name)
  {
   string bg = name + "_BG";
   if(ObjectFind(0, bg) >= 0)
      ObjectDelete(0, bg);
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
  }
//+==================================================================+
//|  Перемещаемые панели (по статье mql5.com/ru/articles/12923)      |
//|  Панель 1 — торговая (кнопка BUY + лот).                         |
//|  Панель 2 — список всех логов советника.                         |
//+------------------------------------------------------------------+
int    g_ptx = 20;                          // базовая X торговой панели
int    g_pty = 80;                          // базовая Y торговой панели
int    g_plx = 0;                           // базовая X лог-панели
int    g_ply = 0;                           // базовая Y лог-панели
double g_pnl_lot = 0.0;                     // рабочий лот для кнопок

void   LogAdd(string msg);                  // вперёд-объявление
void   ExecuteOpenManualTrade();            // вперёд-объявление

//+------------------------------------------------------------------+
//| Создать/обновить прямоугольную метку (подложку)                  |
//+------------------------------------------------------------------+
void Pn_Rect(const string name, const int x, const int y, const int w, const int h,
             const color bg)
  {
   if(ObjectFind(0,name) < 0)
     {
      ObjectCreate(0,name,OBJ_RECTANGLE_LABEL,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,0);
      ObjectSetInteger(0,name,OBJPROP_FILL,true);
      ObjectSetInteger(0,name,OBJPROP_BORDER_TYPE,BORDER_FLAT);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
   ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bg);
  }
//+------------------------------------------------------------------+
//| Создать/обновить текстовую метку                                 |
//+------------------------------------------------------------------+
void Pn_Lbl(const string name, const int x, const int y, const string txt,
            const color c, const int fs=10)
  {
   if(ObjectFind(0,name)<0)
     {
      ObjectCreate(0,name,OBJ_LABEL,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,2);
      ObjectSetString(0,name,OBJPROP_FONT,"Consolas");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,fs);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_COLOR,c);
   ObjectSetString(0,name,OBJPROP_TEXT,txt);
  }
//+------------------------------------------------------------------+
//| Создать/обновить кнопку                                          |
//+------------------------------------------------------------------+
void Pn_Btn(const string name, const int x, const int y, const int w, const int h,
            const string txt, const color bg, const color fg)
  {
   if(ObjectFind(0,name)<0)
     {
      ObjectCreate(0,name,OBJ_BUTTON,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,true);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,false);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,0);
      ObjectSetString(0,name,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,11);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bg);
   ObjectSetInteger(0,name,OBJPROP_COLOR,fg);
   ObjectSetString(0,name,OBJPROP_TEXT,txt);
  }
//+------------------------------------------------------------------+
//| Сборка перемещаемых панелей                                      |
//+------------------------------------------------------------------+
void Panel_Build()
  {
   // Торговая панель (слева сверху)
   g_ptx=20; g_pty=60;
   Pn_Rect(g_pan+"_Title", g_ptx, g_pty, 190, 26, C'47,54,153');
   Pn_Lbl(g_pan+"_TitleTxt", g_ptx+8, g_pty+5, "RNF Filter Trading (BUY)", clrWhite, 11);
   Pn_Rect(g_pan+"_Border", g_ptx+2, g_pty+27, 186, 62, C'90,90,96');
   Pn_Lbl(g_pan+"_LotTxt", g_ptx+12, g_pty+36, "Lot:", clrSilver, 11);
   g_pnl_lot=Inp_LotSize;
   Pn_Lbl(g_pan+"_LotVal", g_ptx+46, g_pty+34,
          DoubleToString(NormalizeLot(g_pnl_lot),2), clrWhite, 12);
   Pn_Btn(g_pan+"_Buy", g_ptx+12, g_pty+64, 170, 28, "BUY", C'0,128,0', clrWhite);

   // Лог-панель (справа)
   g_plx=310; g_ply=60;
   Pn_Rect(g_logp+"_Title", g_plx, g_ply, 340, 26, C'64,64,80');
   Pn_Lbl(g_logp+"_TitleTxt", g_plx+8, g_ply+3, "RNNFilter Logs", clrWhite, 11);
   Pn_Rect(g_logp+"_Border", g_plx+2, g_ply+27, 336, 260, C'70,70,76');
   Pn_Lbl(g_logp+"_Body", g_plx+8, g_ply+32, "Лог пуст", clrSilver, 9);

   // Заголовки делаем перетаскиваемыми (управляющий элемент)
   ObjectSetInteger(0,g_pan+"_Title",OBJPROP_SELECTABLE,true);
   ObjectSetInteger(0,g_pan+"_Title",OBJPROP_HIDDEN,false);
   ObjectSetInteger(0,g_logp+"_Title",OBJPROP_SELECTABLE,true);
   ObjectSetInteger(0,g_logp+"_Title",OBJPROP_HIDDEN,false);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
//| Уничтожение графических объектов панелей                         |
//+------------------------------------------------------------------+
void Panel_Destroy()
  {
   string del[] = {"_Title","_TitleTxt","_Border","_LotTxt","_LotVal","_Buy"};
   for(int i=0;i<ArraySize(del);i++){ DeleteObj(g_pan+del[i]); }
   for(int i=0;i<ArraySize(del);i++){ DeleteObj(g_logp+del[i]); }
   DeleteObj(g_logp+"_Body");
  }
//+------------------------------------------------------------------+
//| Удалить объект по имени (обёртка)                              |
//+------------------------------------------------------------------+
void DeleteObj(const string nm)
  {
   if(ObjectFind(0,nm)>=0) ObjectDelete(0,nm);
  }
//+------------------------------------------------------------------+
//| Добавить запись в лог и обновить лог-панель                      |
//+------------------------------------------------------------------+
void LogAdd(string msg)
  {
   // Логи пишем только в журнал терминала, на график не выводим
   Print(msg);
  }
//+------------------------------------------------------------------+
//| Обновить информационный блок панели RNN_Log                      |
//| 1: порог сети (из параметров)                                    |
//| 2: confidence BUY                                                |
//| 3: решение PASS/BUY (просто информация)                          |
//+------------------------------------------------------------------+
void Panel_LogInfo()
  {
   string buy_txt  = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_buy)
                                                            : "0.0000 (NN off)";
   string h4_txt = StringFormat("H4: %s (ADX %.1f)", H4_TrendFilter_GetState(), H4_TrendFilter_GetADX());
   string info = StringFormat("Порог   BUY=%.2f", Inp_BuyConfThreshold) + "\n"
               + StringFormat("Conf BUY = %s", buy_txt) + "\n"
               + h4_txt + "\n"
               + StringFormat("Решение: %s", g_last_decision);
   Pn_Lbl(g_logp+"_Body", g_plx+8, g_ply+32, info, clrWhite, 11);
  }
//+------------------------------------------------------------------+
//| Обработка кликов и перетаскивания панелей                        |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   // --- Перетаскивание панелей мышью (Rectangle Label не тянется нативно) ---
   if(id==CHARTEVENT_MOUSE_MOVE)
     {
      int mx=(int)lparam;
      int my=(int)dparam;
      bool left=((StringToInteger(sparam) & 1)!=0);   // левая кнопка нажата

      if(!left)
        {
         g_drag_pan=false;
         g_drag_log=false;
         return;
        }

      if(!g_drag_pan && !g_drag_log)
        {
         // начало перетаскивания: курсор в заголовной полосе (190x26)
         if(mx>=g_ptx && mx<=g_ptx+190 && my>=g_pty && my<=g_pty+26)
           { g_drag_pan=true; g_drag_dx=mx-g_ptx; g_drag_dy=my-g_pty; }
         else if(mx>=g_plx && mx<=g_plx+340 && my>=g_ply && my<=g_ply+26)
           { g_drag_log=true; g_drag_dx=mx-g_plx; g_drag_dy=my-g_ply; }
        }

      if(g_drag_pan)
        {
         int nx=mx-g_drag_dx, ny=my-g_drag_dy;
         ShiftPanel(g_pan, nx-g_ptx, ny-g_pty);
         g_ptx=nx; g_pty=ny;
         ChartRedraw(0);
        }
      else if(g_drag_log)
        {
         int nx=mx-g_drag_dx, ny=my-g_drag_dy;
         ShiftPanel(g_logp, nx-g_plx, ny-g_ply);
         g_plx=nx; g_ply=ny;
         ChartRedraw(0);
        }
     }
   // --- Клики по кнопкам BUY ---
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      if(sparam==g_pan+"_Buy") ExecuteOpenManualTrade();
     }
  }
//+------------------------------------------------------------------+
//| Сдвинуть все объекты панели с общим префиксом на dx,dy          |
//+------------------------------------------------------------------+
void ShiftPanel(const string prefix,const int dx,const int dy)
  {
   string parts[] = {"_Title","_TitleTxt","_Border","_LotTxt","_LotVal","_Buy","_Body"};
   for(int i=0;i<ArraySize(parts);i++)
     {
      string nm=prefix+parts[i];
      if(ObjectFind(0,nm)<0) continue;
      int x=(int)ObjectGetInteger(0,nm,OBJPROP_XDISTANCE)+dx;
      int y=(int)ObjectGetInteger(0,nm,OBJPROP_YDISTANCE)+dy;
      ObjectSetInteger(0,nm,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,nm,OBJPROP_YDISTANCE,y);
     }
  }
//+------------------------------------------------------------------+
//| Ручное открытие рыночной сделки по кнопке панели (BUY-only)      |
//+------------------------------------------------------------------+
void ExecuteOpenManualTrade()
  {
   if(HasOpenPosition()) { LogAdd("Ручной вход: уже есть открытая позиция"); return; }
   if(!Inp_Enabled)      { LogAdd("Ручной вход заблокирован"); return; }
   double lot=NormalizeLot(g_pnl_lot);
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double sl=0.0, tprice=0.0;
   if(Inp_SL_Points>0)
     {
      sl=ask-Inp_SL_Points*_Point;
      tprice=(Inp_TP_Points>0) ? ask+tp_price : 0.0;
     }
   bool ok=false;
   ok=trade.Buy(lot,_Symbol,ask,sl,tprice,"RNN_WO_Man");
   if(ok)
     {
      ulong ticket=trade.ResultOrder();
      LogAdd(StringFormat("Ручная сделка BUY #%I64u лот %.2f",ticket,lot));
      // добавить в отслеживание как обычную сделку без НС
      AddDeal(ticket, 0, TimeCurrent(), ask, sl, tprice);
     }
   else LogAdd(StringFormat("Ошибка ручной сделки: %d",GetLastError()));
   UpdateStatus();
  }
//+------------------------------------------------------------------+
//| Индикация тренда H1 (направление + сила) на графике              |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Строка состояния (panels)                                        |
//+------------------------------------------------------------------+
void UpdateStatus()
  {
   string conf_buy  = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_buy)
                                                              : StringFormat("%.4f (NN off)", 0.0);

   string pos = "Сделок: нет";
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Inp_Magic) continue;
      string stype = "BUY";
      double vol    = PositionGetDouble(POSITION_VOLUME);
      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      pos = StringFormat("Сделка: %s  vol=%.2f  profit=%.2f", stype, vol, profit);
      break;
     }

   string mode_txt = (Inp_TradeWOIndicators) ?
                      "Режим: БЕЗ индикаторов - только по доверию сети" :
                      "Режим: индикаторные сигналы + подтверждение сети";
   string monitor = "RNNFilter   (BUY-only, одна активная сделка)\n"
           
           + StringFormat("BUY  сеть:  conf = %s   (порог %.2f)\n", conf_buy, Inp_BuyConfThreshold)
                    + mode_txt + "\n"
                    + "Условие BUY: BUY-сеть уверена + тренд H1 подтверждён\n"
                    + pos;

   // --- Отрисовка панели тренда H1 на графике ---
   
  }
//+------------------------------------------------------------------+
