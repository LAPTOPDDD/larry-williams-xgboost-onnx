import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# =====================================================================
# НАСТРОЙКИ (Меняй здесь)
# =====================================================================
DATA_FILE = "data/EURUSD_H1_20100101_20260120.csv"  # Имя файла с данными (только один инструмент)
TRAIN_BARS = 500                 # Размер обучающей выборки (баров)
TEST_BARS = 500                  # Размер тестовой выборки (баров)
RR_RATIO = 1.5                   # Риск-профит (например, 1.5 значит риск 1 к 1.5 прибыли)
HORIZON = 20                     # Глубина прогноза (на скольких барах ждем TP/SL)
RISK_PER_TRADE = 0.01            # Риск на сделку (1% от депозита)
INITIAL_BALANCE = 10000.0        # Стартовый депозит
# =====================================================================

def build_features(df):
    df.columns = [c.strip().upper() for c in df.columns]
    df['is_outside_bull'] = ((df['LOW'] < df['LOW'].shift(1)) & (df['HIGH'] > df['HIGH'].shift(1)) & (df['CLOSE'] > df['HIGH'].shift(1))).astype(int)
    df['is_outside_bear'] = ((df['HIGH'] > df['HIGH'].shift(1)) & (df['LOW'] < df['LOW'].shift(1)) & (df['CLOSE'] < df['LOW'].shift(1))).astype(int)
    
    tr = np.maximum(df["HIGH"] - df["LOW"], np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df["HIGH"] - df["LOW"] + 1e-9)
    df["REL_RANGE"] = (df["HIGH"] - df["LOW"]) / (df["HIGH"].rolling(10).max() - df["LOW"].rolling(10).min() + 1e-9)
    df['DAY'] = pd.to_datetime(df['DATETIME']).dt.dayofweek
    df['HOUR'] = pd.to_datetime(df['DATETIME']).dt.hour
    return df

def label_profit(df):
    df['target_profit'] = 0.0
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    bull, bear = df['is_outside_bull'].values, df['is_outside_bear'].values
    for i in range(len(df) - HORIZON):
        entry_price = o[i+1]
        if bull[i] == 1:
            sl = l[i]
            tp = entry_price + (entry_price - sl) * RR_RATIO
            if entry_price != sl:
                for j in range(i+1, i+HORIZON):
                    if l[j] <= sl: 
                        df.at[df.index[i], 'target_profit'] = -(entry_price - sl); break 
                    if h[j] >= tp:
                        df.at[df.index[i], 'target_profit'] = (tp - entry_price); break
        elif bear[i] == 1:
            sl = h[i]
            tp = entry_price - (sl - entry_price) * RR_RATIO
            if entry_price != sl:
                for j in range(i+1, i+HORIZON):
                    if h[j] >= sl: 
                        df.at[df.index[i], 'target_profit'] = -(sl - entry_price); break
                    if l[j] <= tp:
                        df.at[df.index[i], 'target_profit'] = (entry_price - tp); break
    return df

def run():
    print("1. Загрузка данных...")
    if not os.path.exists(DATA_FILE):
        print(f"Ошибка: Файл {DATA_FILE} не найден!"); return

    df = pd.read_csv(DATA_FILE, sep=None, engine='python')
    df = build_features(df)
    df = label_profit(df)
    
    # Берем только нужное количество баров для теста
    start_idx = max(0, len(df) - (TRAIN_BARS + TEST_BARS))
    df = df.iloc[start_idx:].reset_index(drop=True)
    
    features = ["BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR", "DAY", "HOUR"]
    df = df.dropna(subset=features + ['target_profit']).reset_index(drop=True)
    
    if len(df) < TRAIN_BARS + TEST_BARS:
        print(f"Недостаточно данных. Нужно {TRAIN_BARS + TEST_BARS}, есть {len(df)}"); return

    df_train = df.iloc[:TRAIN_BARS]
    df_test = df.iloc[TRAIN_BARS : TRAIN_BARS + TEST_BARS].reset_index(drop=True)
    
    print(f"Обучение на {len(df_train)} барах. Тест на {len(df_test)} барах.")

    # --- ОБУЧЕНИЕ ---
    X_train = df_train[features]
    y_train = df_train['target_profit']
    
    # Одна простая модель (XGBoost)
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    print("Модель обучена.")

    # --- БЭКТЕСТ ---
    equity = [INITIAL_BALANCE]
    dates = []
    in_pos = 0
    entry_p, sl_p, tp_p, pos_size = 0, 0, 0, 0
    
    print("Запуск бэктеста...")
    for i in range(len(df_test) - 1):
        current = df_test.iloc[i]
        next_bar = df_test.iloc[i+1]
        
        # Закрытие
        if in_pos != 0:
            pl = 0
            close_trade = False
            if in_pos == 1:
                if next_bar['LOW'] <= sl_p:
                    pl = -pos_size * abs(entry_p - sl_p); close_trade = True
                elif next_bar['HIGH'] >= tp_p:
                    pl = pos_size * abs(tp_p - entry_p); close_trade = True
            elif in_pos == -1:
                if next_bar['HIGH'] >= sl_p:
                    pl = -pos_size * abs(sl_p - entry_p); close_trade = True
                elif next_bar['LOW'] <= tp_p:
                    pl = pos_size * abs(entry_p - tp_p); close_trade = True
            
            if close_trade:
                equity[-1] += pl
                in_pos = 0

        # Открытие
        if in_pos == 0:
            X_curr = df_test.iloc[i:i+1][features]
            pred = model.predict(X_curr)[0]
            
            if current['is_outside_bull'] == 1 and pred > 0:
                in_pos = 1; entry_p = next_bar['OPEN']; sl_p = current['LOW']
                if entry_p > sl_p: tp_p = entry_p + (entry_p - sl_p) * RR_RATIO; pos_size = (equity[-1] * RISK_PER_TRADE) / abs(entry_p - sl_p)
                else: in_pos = 0
            elif current['is_outside_bear'] == 1 and pred < 0:
                in_pos = -1; entry_p = next_bar['OPEN']; sl_p = current['HIGH']
                if sl_p > entry_p: tp_p = entry_p - (sl_p - entry_p) * RR_RATIO; pos_size = (equity[-1] * RISK_PER_TRADE) / abs(sl_p - entry_p)
                else: in_pos = 0

        equity.append(equity[-1])
        dates.append(pd.to_datetime(current['DATETIME']))

    # --- ГРАФИК ---
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity, color='blue', linewidth=1.5)
    plt.title(f'Equity Curve: {os.path.basename(DATA_FILE)} | RR={RR_RATIO} | Horizon={HORIZON}')
    plt.xlabel('Time'); plt.ylabel('Balance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()