import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RR = 1.5
RISK_PCT = 0.01
THRESHOLD = 0.6
START, END = '2020-01-24', '2025-12-03'

FEATURES = [
    "BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR",
    "RELATIVE_ATR", "DAY", "HOUR", "VOL_CHANGE", "PREV_DIR"
]

def build_features(df):
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

def backtest_xau(model, threshold, use_argmax=False):
    f = os.path.join(os.path.dirname(__file__), 'data', 'XAUUSD_H4_20100101_20260120.csv')
    df = pd.read_csv(f, sep='\t')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df[(df['DATETIME'] >= START) & (df['DATETIME'] <= END)].reset_index(drop=True)
    df = build_features(df).reset_index(drop=True)

    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    dt = pd.to_datetime(df['DATETIME']).values
    sig = ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values
    sig_idx = np.where(sig)[0]

    feat = df.loc[sig, FEATURES]
    if use_argmax:
        ar = model.predict(feat)
    else:
        ar = None
    proba = model.predict_proba(feat)  # [P(0), P(Buy), P(Sell)]

    trades, eq = [], 1.0
    eq_series = [(pd.Timestamp(dt[0]), 1.0)]
    open_until = -1
    for k, i in enumerate(sig_idx):
        if i + 1 >= len(df) or i <= open_until:
            continue
        if use_argmax:
            if ar[k] == 1:
                side = 1
            elif ar[k] == 2:
                side = 2
            else:
                continue
        else:
            pb, ps = proba[k, 1], proba[k, 2]
            if pb > threshold:
                side = 1
            elif ps > threshold:
                side = 2
            else:
                continue
        entry_bar = i + 1
        entry = o[entry_bar]
        if side == 1:
            sl, r = l[i], entry - l[i]
            tp = entry + r * RR
        else:
            sl = h[i]; r = sl - entry
            tp = entry - r * RR
        if r <= 0:
            continue
        exit_price, ex_reason, exit_bar = None, None, None
        for j in range(entry_bar, len(df)):
            if side == 1:
                if l[j] <= sl:
                    exit_price, ex_reason, exit_bar = sl, 'SL', j; break
                if h[j] >= tp:
                    exit_price, ex_reason, exit_bar = tp, 'TP', j; break
            else:
                if h[j] >= sl:
                    exit_price, ex_reason, exit_bar = sl, 'SL', j; break
                if l[j] <= tp:
                    exit_price, ex_reason, exit_bar = tp, 'TP', j; break
        if exit_price is None:
            continue
        pnl_r = (exit_price - entry) / r if side == 1 else (entry - exit_price) / r
        eq *= (1 + RISK_PCT * pnl_r)
        open_until = exit_bar
        trades.append(dict(entry_dt=pd.Timestamp(dt[entry_bar]), side='BUY' if side == 1 else 'SELL',
                           pnl_r=pnl_r, reason=ex_reason, bars_held=exit_bar - entry_bar))
        eq_series.append((pd.Timestamp(dt[exit_bar]), eq))

    tr = pd.DataFrame(trades)
    n = len(tr)
    if n == 0:
        return None, None, None
    wins = (tr['pnl_r'] > 0).sum()
    gp = tr.loc[tr['pnl_r'] > 0, 'pnl_r'].sum()
    gl = -tr.loc[tr['pnl_r'] < 0, 'pnl_r'].sum()
    eq_df = pd.DataFrame(eq_series, columns=['ts', 'eq'])
    peak = eq_df['eq'].cummax()
    stats = dict(trades=n, wins=int(wins), losses=int(n - wins),
                 win_rate=wins / n, profit_factor=(gp / gl if gl else np.inf),
                 avg_r=tr['pnl_r'].mean(), total_r=tr['pnl_r'].sum(),
                 final_eq=eq_df['eq'].iloc[-1], max_dd=(eq_df['eq'] / peak - 1).min(),
                 buys=int((tr['side'] == 'BUY').sum()), sells=int((tr['side'] == 'SELL').sum()))
    return eq_df, stats, tr

def main():
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'larry_model.pkl'))
    print(f"XAUUSD H4  |  {START} - {END}  |  риск {RISK_PCT:.0%}/сделку\n")
    print(f"{'thr':>4} {'trades':>6} {'buy':>6} {'sell':>6} {'win%':>6} {'PF':>6} {'avgR':>7} {'totR':>8} {'final':>9} {'maxDD':>8}")
    rows = []
    for thr in [0.5, 0.6, 0.7, 0.8]:
        eq, st, tr = backtest_xau(model, thr)
        rows.append((thr, st, tr))
        if st is None:
            print(f"{thr:.1f} no trades"); continue
        print(f"{thr:.1f} {st['trades']:6d} {st['buys']:6d} {st['sells']:6d} "
              f"{st['win_rate']:6.1%} {st['profit_factor']:6.2f} {st['avg_r']:7.3f} "
              f"{st['total_r']:8.1f} {st['final_eq']:9.2f}x {st['max_dd']:8.1%}")

    # argmax (как в моём первом прогоне) для сравнения
    eq_a, st_a, tr_a = backtest_xau(model, 0.0, use_argmax=True)
    print(f"argm {st_a['trades']:6d} {st_a['buys']:6d} {st_a['sells']:6d} "
          f"{st_a['win_rate']:6.1%} {st_a['profit_factor']:6.2f} {st_a['avg_r']:7.3f} "
          f"{st_a['total_r']:8.1f} {st_a['final_eq']:9.2f}x {st_a['max_dd']:8.1%}")

    thr, st, tr = rows[1]  # 0.6 по умолчанию в EA

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(eq_a['ts'], eq_a['eq'], lw=1.0, label='argmax (первый прогон)', alpha=0.6)
    ax[0].plot(eq['ts'], eq['eq'], lw=1.4, label=f'threshold={thr} (как в EA)', color='#d62728')
    ax[0].axhline(1, color='gray', lw=0.7, ls='--')
    ax[0].set_yscale('log'); ax[0].grid(alpha=0.3); ax[0].legend()
    dd = eq['eq'] / eq['eq'].cummax() - 1
    ax[1].fill_between(eq['ts'], dd, 0, color='red', alpha=0.4); ax[1].grid(alpha=0.3)
    tt = tr['entry_dt'].min().date(), tr['entry_dt'].max().date()
    fig.suptitle(f"XAUUSD H4 | threshold={thr} | trades={st['trades']} win%={st['win_rate']:.1%} "
                 f"PF={st['profit_factor']:.2f} totR={st['total_r']:.1f} final={st['final_eq']:.2f}x "
                 f"maxDD={st['max_dd']:.1%}\n{tt[0]} - {tt[1]}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(os.path.dirname(__file__), 'equity_xau_h4_ea.png'), dpi=130)
    plt.close(fig)
    print("\nPlot: equity_xau_h4_ea.png")

    print("\nПример сделок:")
    print(tr.head(10).to_string())

if __name__ == '__main__':
    main()