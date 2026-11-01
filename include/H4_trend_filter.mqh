//+------------------------------------------------------------------+
//|                                              H4_trend_filter.mqh  |
//|   Единый модуль фильтра тренда H4 (ADX/+DI/-DI)                  |
//|   Используется в PrepareData.mq5 и RNNFilter_Gold_M5.mq5         |
//|   Позволяет синхронизировать логику тренда между сборкой        |
//|   данных и торговым советником.                                  |
//+------------------------------------------------------------------+
#ifndef H4_TREND_FILTER_MQH
#define H4_TREND_FILTER_MQH

//+------------------------------------------------------------------+
//| Глобальные переменные фильтра тренда H4                          |
//+------------------------------------------------------------------+
int            g_h4_handle_adx     = INVALID_HANDLE;  // Хендл ADX на H4
string         g_h4_trend_state    = "FLAT";          // BULL / BEAR / FLAT
datetime       g_h4_last_bar       = 0;               // Последний обработанный бар H4
double         g_h4_last_adx       = 0.0;             // Последнее значение ADX (для мониторинга)

//+------------------------------------------------------------------+
//| Инициализация H4-фильтра (вызывается из OnInit)                  |
//+------------------------------------------------------------------+
bool H4_TrendFilter_Init(int adx_period, double adx_threshold)
  {
   // Сохраняем параметры в глобальные (если нужны для функций)
   // Хотя они могут быть локальными в модуле

   g_h4_handle_adx = iADX(_Symbol, PERIOD_H4, adx_period);
   if(g_h4_handle_adx == INVALID_HANDLE)
     {
      PrintFormat("H4_TrendFilter_Init: ошибка создания ADX (период %d, код %d)",
                  adx_period, GetLastError());
      return false;
     }

   PrintFormat("H4_TrendFilter: инициализирован (ADX период %d, порог %.0f)",
               adx_period, adx_threshold);
   return true;
  }

//+------------------------------------------------------------------+
//| Деинициализация H4-фильтра (вызывается из OnDeinit)              |
//+------------------------------------------------------------------+
void H4_TrendFilter_Deinit()
  {
   if(g_h4_handle_adx != INVALID_HANDLE)
     {
      IndicatorRelease(g_h4_handle_adx);
      g_h4_handle_adx = INVALID_HANDLE;
     }
  }

//+------------------------------------------------------------------+
//| Обновление состояния тренда H4 (вызывается из OnTick)            |
//| Параметры:                                                       |
//|   adx_threshold - порог для определения силы тренда (классика=25)|
//+------------------------------------------------------------------+
void H4_TrendFilter_Update(double adx_threshold)
  {
   datetime h4_bar = iTime(_Symbol, PERIOD_H4, 0);
   if(h4_bar == g_h4_last_bar)
      return;  // Тот же бар H4 - ничего не делаем

   g_h4_last_bar = h4_bar;

   double adx[], pdi[], mdi[];
   if(CopyBuffer(g_h4_handle_adx, 0, 1, 1, adx)    == 1 &&
      CopyBuffer(g_h4_handle_adx, 1, 1, 1, pdi)  == 1 &&
      CopyBuffer(g_h4_handle_adx, 2, 1, 1, mdi) == 1)
     {
      g_h4_last_adx = adx[0];  // Сохраняем для мониторинга

      if(pdi[0] > mdi[0] && adx[0] >= adx_threshold)
         g_h4_trend_state = "BULL";
      else if(mdi[0] > pdi[0] && adx[0] >= adx_threshold)
         g_h4_trend_state = "BEAR";
      else
         g_h4_trend_state = "FLAT";
     }
  }

//+------------------------------------------------------------------+
//| Получить текущее состояние тренда                                |
//+------------------------------------------------------------------+
string H4_TrendFilter_GetState()
  {
   return g_h4_trend_state;
  }

//+------------------------------------------------------------------+
//| Получить последнее значение ADX                                  |
//+------------------------------------------------------------------+
double H4_TrendFilter_GetADX()
  {
   return g_h4_last_adx;
  }

//+------------------------------------------------------------------+
//| Проверить, бычий ли тренд (для условий входа)                   |
//+------------------------------------------------------------------+
bool H4_TrendFilter_IsBullish()
  {
   return (g_h4_trend_state == "BULL");
  }

//+------------------------------------------------------------------+
//| Проверить, медвежий ли тренд                                     |
//+------------------------------------------------------------------+
bool H4_TrendFilter_IsBearish()
  {
   return (g_h4_trend_state == "BEAR");
  }

//+------------------------------------------------------------------+
//| Проверить, флет                                                   |
//+------------------------------------------------------------------+
bool H4_TrendFilter_IsFlat()
  {
   return (g_h4_trend_state == "FLAT");
  }

//+------------------------------------------------------------------+
//| Отрисовка метки тренда на графике (опционально, для визуализации)|
//| Используется в PrepareData.mq5 для мониторинга                  |
//+------------------------------------------------------------------+
void H4_TrendFilter_DrawLabel()
  {
   string nm = "H4_Trend_Label";
   if(ObjectFind(0, nm) < 0)
     {
      ObjectCreate(0, nm, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, nm, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, nm, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, nm, OBJPROP_YDISTANCE, 30);
      ObjectSetInteger(0, nm, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, nm, OBJPROP_HIDDEN, true);
      ObjectSetString(0, nm, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, nm, OBJPROP_FONTSIZE, 12);
      ObjectSetInteger(0, nm, OBJPROP_ZORDER, 5);
     }

   color c = (g_h4_trend_state == "BULL") ? clrLime :
             (g_h4_trend_state == "BEAR") ? clrRed : clrDodgerBlue;
   ObjectSetInteger(0, nm, OBJPROP_COLOR, c);
   ObjectSetString(0, nm, OBJPROP_TEXT, "TREND(H4): " + g_h4_trend_state);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| Удалить визуальную метку с графика                               |
//+------------------------------------------------------------------+
void H4_TrendFilter_DeleteLabel()
  {
   string nm = "H4_Trend_Label";
   if(ObjectFind(0, nm) >= 0)
      ObjectDelete(0, nm);
  }

#endif
