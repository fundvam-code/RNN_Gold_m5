//+------------------------------------------------------------------+
//|                                              RNN_Scaler.mqh      |
//|   Pre-normalized NPZ input (RNN_gold).                         |
//|   Generated from Optuna best params: Final Training.           |
//+------------------------------------------------------------------+
#ifndef RNN_SCALER_MQH
#define RNN_SCALER_MQH

#define NN_BUY_SEQ_LEN      12
#define NN_SELL_SEQ_LEN     12
#define NN_MAX_SEQ_LEN      12
#define NN_FEATURES         12
#define NN_NORM_STD_EPS     1e-06

#define NN_BUY_NORM_WINDOW      20
#define NN_SELL_NORM_WINDOW     20
#define NN_MAX_NORM_WINDOW      20

#define NN_SEQ_LEN          NN_BUY_SEQ_LEN

#endif // RNN_SCALER_MQH
