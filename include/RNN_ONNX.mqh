//+------------------------------------------------------------------+
//|                                              RNN_ONNX.mqh        |
//|   Подключение GRU-модели BUY (ONNX) к советнику RNNFilter.       |
//|                                                                  |
//|   Контракт v2 (PrepareData v2 / feature_pipeline v2 / rnn_learn):|
//|     - 12 признаков на бар: open, high, low, close, direction,   |
//|       ema8, ema21, rsi, atr, bb_upper, bb_mid, bb_lower          |
//|     - окно нормализации: 20 баров (NN_*_NORM_WINDOW), в сеть     |
//|       идут первые NN_*_SEQ_LEN баров окна (для BUY = 18);        |
//|     - нормализация: row-local z-score (ddof=0) по окну, канал    |
//|       direction (индекс 4) НЕ нормализуется;                     |
//|     - вход ONNX: float32 [1, SEQ_LEN, 12], выход [1, 2] (логиты),|
//|       P1 = уверенность в классе "прибыльной" сделки (softmax).   |
//|                                                                  |
//|   Режим BUY-only: SELL-модель опциональна (если файл задан и    |
//|   загрузился). Советник использует только BUY-сеть.             |
//+------------------------------------------------------------------+
#ifndef RNN_ONNX_MQH
#define RNN_ONNX_MQH

#include "include/RNN_Scaler.mqh"

// Класс "прибыльной" сделки в выходе модели (label [1,0] -> класс 1).
#define NN_PROFIT_CLASS   1

//+------------------------------------------------------------------+
//| Окна нормализации: контракт v2 = 20 баров. Если скалер их не     |
//| определяет (версия из rnn_learn), задаём здесь как fallback.    |
//+------------------------------------------------------------------+
#ifndef NN_BUY_NORM_WINDOW
#define NN_BUY_NORM_WINDOW   20
#endif
#ifndef NN_SELL_NORM_WINDOW
#define NN_SELL_NORM_WINDOW  20
#endif
#ifndef NN_MAX_NORM_WINDOW
#define NN_MAX_NORM_WINDOW   20
#endif

#define NN_NUM_CLASSES    2
int g_nn_seq_len[2]    = {NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN};
int g_nn_input_size[2] = {NN_BUY_SEQ_LEN * NN_FEATURES, NN_SELL_SEQ_LEN * NN_FEATURES};

//--- Хендлы ONNX-сессий (0 = BUY, 1 = SELL), SELL опциональна ---
long   g_onnx_handle[2] = {INVALID_HANDLE, INVALID_HANDLE};
string g_onnx_file[2];            // имена загруженных ONNX-файлов (для лога)
bool   g_onnx_ready     = false;

//--- Формы тензоров (заполняются в NN_ONNX_ApplyConfig) ---
long   g_nn_input_shape_buy[3];
long   g_nn_input_shape_sell[3];
long   g_nn_output_shape[2] = {1, NN_NUM_CLASSES};

//--- Буферы для inference (модель float32 -> данные float32) ---
float  g_nn_input_buy[];
float  g_nn_input_sell[];
float  g_nn_output[NN_NUM_CLASSES];
double g_nn_raw_buy[];
double g_nn_raw_sell[];

//--- Каналы z-нормализации: ВСЕ, кроме direction (индекс 4) ---
int g_nn_norm_mask[NN_FEATURES - 1] = {0,1,2,3,5,6,7,8,9,10,11};
bool g_nn_norm_flag[NN_FEATURES];

//+------------------------------------------------------------------+
//| Флаг нормализации по каналу (заполняется один раз)               |
//+------------------------------------------------------------------+
void NN_BuildNormMask()
  {
   ArrayInitialize(g_nn_norm_flag, false);
   for(int i = 0; i < ArraySize(g_nn_norm_mask); i++)
      g_nn_norm_flag[g_nn_norm_mask[i]] = true;
  }

//+------------------------------------------------------------------+
//| Применение параметров конфига (RNN_Scaler.mqh) к буферам/формам  |
//+------------------------------------------------------------------+
bool NN_ONNX_ApplyConfig()
  {
   NN_BuildNormMask();

   int seq[2]    = {NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN};
   int window[2] = {NN_BUY_NORM_WINDOW, NN_SELL_NORM_WINDOW};
   for(int d = 0; d < 2; d++)
     {
      if(seq[d] > 0 && (seq[d] < 1 || seq[d] > window[d]))
        {
         PrintFormat("ONNX[%s]: seq_len=%d несовместим с norm_window=%d",
                     d == 0 ? "BUY" : "SELL", seq[d], window[d]);
         return(false);
        }
      g_nn_seq_len[d] = seq[d];
      g_nn_input_size[d] = seq[d] * NN_FEATURES;
     }
   if(!ArrayResize(g_nn_raw_buy, NN_BUY_NORM_WINDOW * NN_FEATURES) ||
      !ArrayResize(g_nn_raw_sell, NN_SELL_NORM_WINDOW * NN_FEATURES) ||
      !ArrayResize(g_nn_input_buy, g_nn_input_size[0]) ||
      !ArrayResize(g_nn_input_sell, MathMax(g_nn_input_size[1], 1)))
      return(false);
   g_nn_input_shape_buy[0] = 1; g_nn_input_shape_buy[1] = seq[0]; g_nn_input_shape_buy[2] = NN_FEATURES;
   g_nn_input_shape_sell[0] = 1; g_nn_input_shape_sell[1] = seq[1]; g_nn_input_shape_sell[2] = NN_FEATURES;
   PrintFormat("ONNX: BUY seq=%d window=%d, SELL seq=%d window=%d, features=%d",
               seq[0], window[0], seq[1], window[1], NN_FEATURES);
   return(true);
  }

//+------------------------------------------------------------------+
//| Чтение файла модели в буфер и создание сессии из буфера         |
//| (работает и на графике, и в тестере стратегий)                  |
//+------------------------------------------------------------------+
long NN_OnnxLoadFromBuffer(const string model_path, const int direction)
  {
   string tag = (direction == 0) ? "BUY" : "SELL";

   ResetLastError();
   int file = FileOpen(model_path, FILE_READ | FILE_BIN);
   if(file == INVALID_HANDLE)
      file = FileOpen(model_path, FILE_READ | FILE_BIN | FILE_COMMON);
   if(file == INVALID_HANDLE)
     {
      PrintFormat("ONNX[%s]: не удалось открыть файл '%s' (код %d)",
                  tag, model_path, GetLastError());
      return(INVALID_HANDLE);
     }

   ulong size = FileSize(file);
   if(size <= 0)
     {
      PrintFormat("ONNX[%s]: файл '%s' пуст", tag, model_path);
      FileClose(file);
      return(INVALID_HANDLE);
     }

   uchar buffer[];
   if(!ArrayResize(buffer, (int)size))
     {
      PrintFormat("ONNX[%s]: недостаточно памяти для '%s' (%I64u байт)",
                  tag, model_path, size);
      FileClose(file);
      return(INVALID_HANDLE);
     }

   if(FileReadArray(file, buffer, 0, (int)size) != (int)size)
     {
      PrintFormat("ONNX[%s]: ошибка чтения '%s' (код %d)",
                  tag, model_path, GetLastError());
      FileClose(file);
      return(INVALID_HANDLE);
     }
   FileClose(file);

   long h = OnnxCreateFromBuffer(buffer, 0);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("ONNX[%s]: OnnxCreateFromBuffer('%s') не удался (код %d), пробуем OnnxCreate...",
                  tag, model_path, GetLastError());
      ResetLastError();
      h = OnnxCreate(model_path, 0);
      if(h == INVALID_HANDLE)
        {
         PrintFormat("ONNX[%s]: OnnxCreate('%s') тоже не удался (код %d).",
                     tag, model_path, GetLastError());
         return(INVALID_HANDLE);
        }
      PrintFormat("ONNX[%s]: загружена через OnnxCreate(файл) '%s'", tag, model_path);
     }

   //--- Явно задаём формы входов/выходов ---
   bool shape_ok = false;
   if(direction == 0)
      shape_ok = OnnxSetInputShape(h, 0, g_nn_input_shape_buy);
   else
      shape_ok = OnnxSetInputShape(h, 0, g_nn_input_shape_sell);
   if(!shape_ok)
      PrintFormat("ONNX[%s]: OnnxSetInputShape (код %d)", tag, GetLastError());
   if(!OnnxSetOutputShape(h, 0, g_nn_output_shape))
      PrintFormat("ONNX[%s]: OnnxSetOutputShape (код %d)", tag, GetLastError());

   //--- Диагностика ---
   long in_count  = OnnxGetInputCount(h);
   long out_count = OnnxGetOutputCount(h);
   string in_name  = (in_count > 0)  ? OnnxGetInputName(h, 0)  : "?";
   string out_name = (out_count > 0) ? OnnxGetOutputName(h, 0) : "?";

   string our_shape = "";
   long shape_value[3];
   shape_value[0] = 1;
   shape_value[1] = g_nn_seq_len[direction];
   shape_value[2] = NN_FEATURES;
   for(int i = 0; i < 3; i++)
      our_shape += (i > 0 ? "x" : "") + IntegerToString(shape_value[i]);

   PrintFormat("ONNX[%s]: '%s' готова (%I64u байт) | вход: '%s' [%s], выход: '%s' 2",
               tag, model_path, size, in_name, our_shape, out_name);

   g_onnx_handle[direction] = h;
   g_onnx_file[direction]   = model_path;
   return(h);
  }

//+------------------------------------------------------------------+
//| Инициализация: BUY-модель обязательна, SELL опциональна.        |
//| SELL-модель не загружается, если файл пуст/не задан ("") или    |
//| равен "-"; при неудаче загрузки SELL продолжаем с BUY (BUY-only).|
//+------------------------------------------------------------------+
bool NN_ONNX_Init(const string buy_model_file, const string sell_model_file)
  {
   g_onnx_ready = false;

   if(!NN_ONNX_ApplyConfig())
      return(false);

   if(NN_OnnxLoadFromBuffer(buy_model_file, 0) == INVALID_HANDLE)
      return(false);

   // --- SELL опциональна (советник BUY-only) ---
   if(sell_model_file == "" || StringCompare(sell_model_file, "-") == 0)
     {
      Print("ONNX: SELL-модель не задана (BUY-only режим)");
     }
   else if(NN_OnnxLoadFromBuffer(sell_model_file, 1) == INVALID_HANDLE)
     {
      Print("ONNX: SELL-модель не загружена — продолжаю в режиме BUY-only");
     }

   g_onnx_ready = true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Деинициализация: освобождение сессий                             |
//+------------------------------------------------------------------+
void NN_ONNX_Deinit()
  {
   for(int i = 0; i < 2; i++)
     {
      if(g_onnx_handle[i] != INVALID_HANDLE)
        {
         OnnxRelease(g_onnx_handle[i]);
         g_onnx_handle[i] = INVALID_HANDLE;
        }
     }
   g_onnx_ready = false;
  }

//+------------------------------------------------------------------+
//| Признаки одного бара (12 фич, контракт PrepareData v2).          |
//| direction = +1 (бычья свеча), иначе -1. Все значения СЫРЫЕ       |
//| (нормализация делается позже по окну).                           |
//+------------------------------------------------------------------+
void NN_FillBarFeatures(const int bar,
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
   out[0]  = open[bar];                            // open
   out[1]  = high[bar];                            // high
   out[2]  = low[bar];                             // low
   out[3]  = close[bar];                           // close
   out[4]  = (close[bar] > open[bar]) ? 1.0 : -1.0;// direction (не нормируется)
   out[5]  = ema8[bar];                            // ema8
   out[6]  = ema21[bar];                           // ema21
   out[7]  = rsi[bar];                             // rsi
   out[8]  = atr[bar];                             // atr
   out[9]  = bb_upper[bar];                        // bb_upper
   out[10] = bb_mid[bar];                          // bb_mid
   out[11] = bb_lower[bar];                        // bb_lower
  }

//+------------------------------------------------------------------+
//| Построение вектора СЫРЫХ признаков (norm_window баров x 12).     |
//| Порядок баров: сигнальный (самый свежий, lag0) -> более старые. |
//| Если истории меньше norm_window — недостающие (старые) бары      |
//| добиваются средним по доступным барам.                           |
//+------------------------------------------------------------------+
void NN_BuildFeatures(const int signal_shift,
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
                      const int norm_window,
                      const int avail_bars,
                      double &raw[])
  {
   double bar_feat[NN_FEATURES];
   double feat_mean[NN_FEATURES];
   ArrayInitialize(feat_mean, 0.0);

   int n_real = avail_bars - signal_shift;
   if(n_real > norm_window)
      n_real = norm_window;
   if(n_real < 1)
      n_real = 1;

   // --- проход 1: средние по реальным барам (для добивки старых) ---
   for(int k = 0; k < n_real; k++)
     {
      NN_FillBarFeatures(signal_shift + k, open, close, high, low,
                         ema8, ema21, rsi, atr,
                         bb_upper, bb_mid, bb_lower, bar_feat);
      for(int f = 0; f < NN_FEATURES; f++)
         feat_mean[f] += bar_feat[f];
     }
   for(int f = 0; f < NN_FEATURES; f++)
      feat_mean[f] /= (double)n_real;

   // --- проход 2: заполнение полного окна (реальные бары + среднее) ---
   int idx = 0;
   for(int k = 0; k < norm_window; k++)
     {
      if(k < n_real)
        {
         NN_FillBarFeatures(signal_shift + k, open, close, high, low,
                            ema8, ema21, rsi, atr,
                            bb_upper, bb_mid, bb_lower, bar_feat);
         for(int f = 0; f < NN_FEATURES; f++)
            raw[idx++] = bar_feat[f];
        }
      else
        {
         for(int f = 0; f < NN_FEATURES; f++)
            raw[idx++] = feat_mean[f];
        }
     }
  }

//+------------------------------------------------------------------+
//| Нормализация контракта v2: row-local z-score (ddof=0) по окну    |
//| для каналов из маски (все, кроме direction=4). Медианная         |
//| коррекция НЕ применяется. В тензор идут первые seq_len баров.    |
//+------------------------------------------------------------------+
void NN_ApplyRollingNorm(const int direction, const double &raw[])
  {
   double mean[NN_FEATURES];
   double std[NN_FEATURES];
   int norm_window = direction == 0 ? NN_BUY_NORM_WINDOW : NN_SELL_NORM_WINDOW;

   ArrayInitialize(mean, 0.0);
   ArrayInitialize(std, 0.0);

   for(int f = 0; f < NN_FEATURES; f++)
     {
      if(!g_nn_norm_flag[f])
         continue;

      double s = 0.0;
      for(int k = 0; k < norm_window; k++)
         s += raw[k * NN_FEATURES + f];
      mean[f] = s / (double)norm_window;

      double s2 = 0.0;
      for(int k = 0; k < norm_window; k++)
        {
         double d = raw[k * NN_FEATURES + f] - mean[f];
         s2 += d * d;
        }
      std[f] = MathSqrt(s2 / (double)norm_window);   // ddof=0
      if(std[f] < NN_NORM_STD_EPS)
         std[f] = NN_NORM_STD_EPS;
     }

   // --- заполнение входного тензора первыми seq_len барами окна ---
   for(int k = 0; k < g_nn_seq_len[direction]; k++)
      for(int f = 0; f < NN_FEATURES; f++)
        {
         int i = k * NN_FEATURES + f;
         double value = g_nn_norm_flag[f] ? ((raw[i] - mean[f]) / std[f]) : raw[i];
         float fv = (float)value;
         if(direction == 0)
            g_nn_input_buy[i] = fv;
         else
            g_nn_input_sell[i] = fv;
        }
  }

//+------------------------------------------------------------------+
//| Прогон модели через ONNX Runtime                                 |
//| direction: 0 = BUY, 1 = SELL                                     |
//| Возвращает softmax-вероятности классов 0 и 1.                    |
//+------------------------------------------------------------------+
bool NN_OnnxRunModel(const int direction, const double &features[],
                     double &prob_class0, double &prob_class1)
  {
   prob_class0 = 0.0;
   prob_class1 = 0.0;

   if(direction < 0 || direction > 1)
      return(false);
   if(g_onnx_handle[direction] == INVALID_HANDLE)
      return(false);
   int norm_window = direction == 0 ? NN_BUY_NORM_WINDOW : NN_SELL_NORM_WINDOW;
   if(g_nn_input_size[direction] <= 0 || ArraySize(features) < norm_window * NN_FEATURES)
     {
      PrintFormat("ONNX[%s]: размер вектора признаков %d < %d (окно нормализации не настроено?)",
                  (direction == 0) ? "BUY" : "SELL",
                  ArraySize(features), norm_window * NN_FEATURES);
      return(false);
     }

   //--- Нормализация контракта v2 (z-score по окну, direction raw) ---
   NN_ApplyRollingNorm(direction, features);

   //--- Инференс (модель float32, данные float32 -> без конвертации) ---
   ResetLastError();
   bool run_ok = false;
   if(direction == 0)
      run_ok = OnnxRun(g_onnx_handle[direction], 0, g_nn_input_buy, g_nn_output);
   else
      run_ok = OnnxRun(g_onnx_handle[direction], 0, g_nn_input_sell, g_nn_output);
   if(!run_ok)
     {
      PrintFormat("ONNX[%s]: ошибка OnnxRun (код %d)",
                  (direction == 0) ? "BUY" : "SELL", GetLastError());
      return(false);
     }

   //--- Softmax ---
   double m = MathMax((double)g_nn_output[0], (double)g_nn_output[1]);
   double e0 = MathExp((double)g_nn_output[0] - m);
   double e1 = MathExp((double)g_nn_output[1] - m);
   double s  = e0 + e1;
   prob_class0 = e0 / s;
   prob_class1 = e1 / s;
   return(true);
  }

//+------------------------------------------------------------------+
//| Уверенность модели в "прибыльной" сделке для направления.        |
//| direction: 0 = BUY, 1 = SELL. profit_class: класс "прибыли"      |
//| или -1.0 при ошибке/отсутствии модели.                           |
//+------------------------------------------------------------------+
double NN_ONNX_Predict(const int direction, const double &features[], const int profit_class = NN_PROFIT_CLASS)
  {
   double p0 = 0.0, p1 = 0.0;
   if(!NN_OnnxRunModel(direction, features, p0, p1))
      return(-1.0);
   double conf = (profit_class == 0) ? p0 : p1;
   PrintFormat("ONNX[%s]: модель '%s' | confidence = %.6f (P0=%.6f P1=%.6f, profit_class=%d)",
               (direction == 0) ? "BUY" : "SELL", g_onnx_file[direction], conf, p0, p1, profit_class);
   return(conf);
  }

//+------------------------------------------------------------------+
//| Готовность модуля (BUY-модель загружена)                         |
//+------------------------------------------------------------------+
bool NN_ONNX_IsReady()
  {
   return(g_onnx_ready);
  }

//+------------------------------------------------------------------+
//| Длина окна НС (баров) — из конфига RNN_Scaler.mqh.               |
//+------------------------------------------------------------------+
int NN_ONNX_GetSeqLen(const int direction = 0)
  {
   return((direction == 0) ? NN_BUY_SEQ_LEN : NN_SELL_SEQ_LEN);
  }


#endif // RNN_ONNX_MQH
