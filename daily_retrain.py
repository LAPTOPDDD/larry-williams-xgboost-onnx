import glob, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

RR = 1.5
HORIZON = 20
THR = 0.6
LOT = 0.01
COMMISSION_USD = 0.07

TEST_START = '2025-12-01'   # последний месяц (с запасом на сделку)
TEST_END = '2026-01-21'

LOOKBACK_SIGNALS = 5000   # скользящее окно: обучаемся на последних N сигнальных барах

FEATURES = ["BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR",
            "RELATIVE_ATR", "DAY", "HOUR", "VOL_CHANGE", "PREV_DIR"]

PARAMS = dict(lr=0.05, depth=6, n=500, sub=0.8, col=0.8, seed=42)  # исходная конфигурация

JPY = {'USDJPY', 'EURJPY', 'GBPJPY'}
DIRECT = {'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'}

def point_size(sym):
    if sym == 'XAUUSD': return 0.01
    if sym in JPY: return 0.001
    return 0.00001

def usd_per_price(sym, px, gbpusd, usdjpy):
    if sym == 'XAUUSD': return 1.0
    if sym in DIRECT:   return 1000.0
    if sym in ('USDJPY', 'USDCAD', 'USDCHF'): return 1000.0 / px
    if sym == 'EURGBP': return 1000.0 * gbpusd
    if sym in ('EURJPY', 'GBPJPY'): return 1000.0 / usdjpy
    return 1000.0

def build_larry_features(df):
    df.columns = [c.strip().upper() for c in df.columns]
    df['is_outside_bull'] = ((df['LOW'] < df['LOW'].shift(1)) &
                             (df['HIGH'] > df['HIGH'].shift(1)) &
                             (df['CLOSE'] > df['HIGH'].shift(1))).astype(int)
    df['is_outside_bear'] = ((df['HIGH'] > df['HIGH'].shift(1)) &
                             (df['LOW'] < df['LOW'].shift(1)) &
                             (df['CLOSE'] < df['LOW'].shift(1))).astype(int)
    tr = np.maximum(df["HIGH"] - df["LOW"],
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)),
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df["HIGH"] - df["LOW"] + 1e-9)
    df["REL_RANGE"] = (df["HIGH"] - df["LOW"]) / (df["HIGH"].rolling(10).max() - df["LOW"].rolling(10).min() + 1e-9)
    df['DAY'] = pd.to_datetime(df['DATETIME']).dt.dayofweek
    df['RELATIVE_ATR'] = (df['HIGH'] - df['LOW']) / df['ATR']
    df['PREV_DIR'] = np.where(df['CLOSE'].shift(1) > df['OPEN'].shift(1), 1, -1)
    df['VOL_CHANGE'] = df['VOLUME'].pct_change()
    df['HOUR'] = pd.to_datetime(df['DATETIME']).dt.hour
    return df

def label_larry_strategy(df):
    df['target'] = 0
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    bull, bear = df['is_outside_bull'].values, df['is_outside_bear'].values
    for i in range(len(df) - HORIZON):
        entry_price = o[i + 1]
        if bull[i] == 1:
            sl = l[i]; tp = entry_price + (entry_price - sl) * RR
            for j in range(i + 1, i + HORIZON):
                if l[j] <= sl: break
                if h[j] >= tp: df.at[df.index[i], 'target'] = 1; break
        elif bear[i] == 1:
            sl = h[i]; tp = entry_price - (sl - entry_price) * RR
            for j in range(i + 1, i + HORIZON):
                if h[j] >= sl: break
                if l[j] <= tp: df.at[df.index[i], 'target'] = 2; break
    return df

def load_universe():
    frames = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv"))):
        name = os.path.basename(f).replace(".csv", "")
        symbol, tf = name.split("_")[0], name.split("_")[1]
        df = pd.read_csv(f, sep='\t')
        df = build_larry_features(df)
        df = label_larry_strategy(df)
        df = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df.attrs['symbol'], df.attrs['tf'] = symbol, tf
        frames.append(df)
    return frames

def sig_mask(df):
    return ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values

def main():
    print(f"ДАILY RETRAIN: обучаем перед каждым торговым днём, торгуем этот день")
    print(f"Тест: {TEST_START} .. {TEST_END} | порог={THR} | лот={LOT} фикс | спред из CSV + комиссия\n")
    frames = load_universe()
    symbols = [f.attrs['symbol'] for f in frames]

    gbpusd = float(frames[symbols.index('GBPUSD')]['CLOSE'].iloc[-1])
    usdjpy = float(frames[symbols.index('USDJPY')]['CLOSE'].iloc[-1])

    # торговые дни тестового окна (по датам входов)
    all_entries = [pd.to_datetime(f['DATETIME']).values for f in frames]
    days = sorted({pd.Timestamp(t).date() for arr in all_entries
                   for t in arr if pd.Timestamp(TEST_START) <= pd.Timestamp(t) < pd.Timestamp(TEST_END)})
    print(f"Торговых дней в тесте: {len(days)}\n")

    # 1) все сигналы вселенной, упорядоченные по времени (для скользящего окна)
    sig_ts, sig_X, sig_y = [], [], []
    for df in frames:
        ts = pd.to_datetime(df['DATETIME']).values
        m = sig_mask(df)
        sig_ts.append(ts[m]); sig_X.append(df.loc[m, FEATURES].values); sig_y.append(df.loc[m, 'target'].values)
    all_ts = np.concatenate(sig_ts)
    order = np.argsort(all_ts)
    all_ts = all_ts[order]
    all_X = np.vstack(sig_X)[order]
    all_y = np.concatenate(sig_y)[order]

    # 2) обучаем модель на последних N сигналах ПЕРЕД каждым торговым днём
    day_models = {}
    for di, D in enumerate(days):
        pos = int(np.searchsorted(all_ts, np.datetime64(D), side='left'))
        lo = max(0, pos - LOOKBACK_SIGNALS)
        X, y = all_X[lo:pos], all_y[lo:pos]
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                              learning_rate=PARAMS['lr'], max_depth=PARAMS['depth'],
                              n_estimators=PARAMS['n'], subsample=PARAMS['sub'],
                              colsample_bytree=PARAMS['col'], tree_method="hist",
                              random_state=PARAMS['seed'])
        m.fit(X, y, sample_weight=compute_sample_weight(class_weight='balanced', y=y))
        day_models[D] = m
        print(f"  день {D.isoformat()}: обучен на {len(X)} сигналах (окно {LOOKBACK_SIGNALS})")

    # 2) торгуем каждый день своей моделью (одна позиция на символ, без лимита по времени)
    trades = []
    for df, sym in zip(frames, symbols):
        h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
        sp = df['SPREAD'].values
        dt = pd.to_datetime(df['DATETIME']).values
        s = sig_mask(df); sig_idx = np.where(s)[0]
        pt = point_size(sym)
        open_until = -1
        for D in days:
            m = day_models[D]
            proba = m.predict_proba(df.loc[s, FEATURES].values)
            for k, i in enumerate(sig_idx):
                if i + 1 >= len(df) or i <= open_until:
                    continue
                entry_ts = dt[i + 1]
                if entry_ts < np.datetime64(D):
                    continue
                if entry_ts >= np.datetime64(D) + np.timedelta64(1, 'D'):
                    break
                pb, ps = proba[k, 1], proba[k, 2]
                if pb > THR: side = 1
                elif ps > THR: side = 2
                else: continue
                eb = i + 1; entry = o[eb]
                if side == 1:
                    sl = l[i]; r = entry - sl
                    if r <= 0: continue
                    tp = entry + r * RR
                else:
                    sl = h[i]
                    if sl <= entry: continue
                    r = sl - entry
                    tp = entry - r * RR
                ex, j = None, eb
                for j in range(i + 1, len(df)):
                    if side == 1:
                        if l[j] <= sl: ex = sl; break
                        if h[j] >= tp: ex = tp; break
                    else:
                        if h[j] >= sl: ex = sl; break
                        if l[j] <= tp: ex = tp; break
                if ex is None: continue
                open_until = j
                gross = (ex - entry) if side == 1 else (entry - ex)
                gross *= usd_per_price(sym, entry, gbpusd, usdjpy)
                cost = sp[eb] * pt * usd_per_price(sym, entry, gbpusd, usdjpy) + COMMISSION_USD
                trades.append(dict(entry=pd.Timestamp(entry_ts), exit=pd.Timestamp(dt[j]),
                                   day=pd.Timestamp(entry_ts).date(), symbol=sym,
                                   gross=gross, cost=cost, net=gross - cost))
    T = pd.DataFrame(trades).sort_values('entry')
    n = len(T)
    win = (T['net'] > 0).mean()
    gp = T.loc[T['net'] > 0, 'net'].sum(); gl = -T.loc[T['net'] < 0, 'net'].sum()
    eq = T.sort_values('exit')['net'].cumsum()
    dd = (eq.cummax() - eq).max()
    days_made = T.groupby('day').apply(lambda x: pd.Series(dict(tr=len(x), net=x['net'].sum())))
    print("\n=== ПОСЛЕДНИЙ МЕСЯЦ: СУТОЧНОЕ ПЕРЕОБУЧЕНИЕ ===")
    print(f"дней с сделками: {(T['day'].nunique())} | сделок: {n} | win%: {win:.1%}")
    print(f"gross: ${T['gross'].sum():.2f} | издержки: ${T['cost'].sum():.2f} | NET: ${T['net'].sum():.2f}")
    print(f"PF(net): {gp/gl if gl else float('inf'):.2f} | MaxDD: ${dd:.2f} | прибыльный день: {((days_made['net']>0).mean()):.0%}")
    print("\nПоследние 10 торговых дней:")
    print(days_made.tail(10).round(2).to_string())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = T.sort_values('exit')['exit'].values
    y = eq.values
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax[0].step(x, y, where='post', color='black', lw=1.8)
    ax[0].axhline(0, color='gray', lw=0.7); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"DAILY RETRAIN | {TEST_START}..{TEST_END} | trades={n} win%={win:.1%} "
                    f"NET=${T['net'].sum():.2f} PF={gp/gl if gl else float('inf'):.2f}")
    ax[1].step(x, eq.cummax() - eq, where='post', color='red', lw=1.2); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "equity_daily_retrain.png"), dpi=130)
    print("Plot: equity_daily_retrain.png")

if __name__ == '__main__':
    main()