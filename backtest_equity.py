import glob, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RR_RATIO = 1.5
HORIZON = 20
RISK_PCT = 0.01  # 1% риска на сделку

FEATURES = [
    "BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR",
    "RELATIVE_ATR", "DAY", "HOUR", "VOL_CHANGE", "PREV_DIR"
]

def build_larry_features(df):
    df = df.copy()
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

def simulate_trades(df, preds):
    """Возвращает список сделок: dict(symbol, tf, entry_dt, side, pnl_r)"""
    trades = []
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    dt = pd.to_datetime(df['DATETIME']).values
    idx = np.arange(len(df))
    n = len(df)
    for i in idx:
        p = preds[i]
        if p not in (1, 2) or i + 1 >= n:
            continue
        entry = o[i + 1]
        if p == 1:  # BUY
            sl = l[i]
            tp = entry + (entry - sl) * RR_RATIO
            r = entry - sl
            exit_price, exit_reason = None, None
            for j in range(i + 1, min(i + 1 + HORIZON, n)):
                if l[j] <= sl:
                    exit_price, exit_reason = sl, 'SL'
                    break
                if h[j] >= tp:
                    exit_price, exit_reason = tp, 'TP'
                    break
            if exit_price is None:
                exit_price, exit_reason = c[min(i + HORIZON, n - 1)], 'TIME'
            pnl_r = (exit_price - entry) / r
        else:  # SELL
            sl = h[i]
            tp = entry - (sl - entry) * RR_RATIO
            r = sl - entry
            exit_price, exit_reason = None, None
            for j in range(i + 1, min(i + 1 + HORIZON, n)):
                if h[j] >= sl:
                    exit_price, exit_reason = sl, 'SL'
                    break
                if l[j] <= tp:
                    exit_price, exit_reason = tp, 'TP'
                    break
            if exit_price is None:
                exit_price, exit_reason = c[min(i + HORIZON, n - 1)], 'TIME'
            pnl_r = (entry - exit_price) / r
        trades.append({
            'symbol': df.attrs.get('symbol'),
            'tf': df.attrs.get('tf'),
            'entry_dt': pd.Timestamp(dt[i + 1]),
            'side': 'BUY' if p == 1 else 'SELL',
            'pnl_r': pnl_r,
            'exit_reason': exit_reason,
        })
    return trades

def run_backtest(model):
    all_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv")))
    all_trades = []
    for f in all_files:
        name = os.path.basename(f)
        parts = name.replace(".csv", "").split("_")
        symbol, tf = parts[0], parts[1]
        df = pd.read_csv(f, sep='\t')
        df.attrs['symbol'], df.attrs['tf'] = symbol, tf
        df = build_larry_features(df)
        df = df.dropna().reset_index(drop=True)
        sig = (df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)
        if not sig.any():
            continue
        feats = df.loc[sig, FEATURES]
        preds_full = np.zeros(len(df), dtype=int)
        preds_full[sig.values] = model.predict(feats)
        trades = simulate_trades(df, preds_full)
        all_trades.extend(trades)
        print(f"  {name:40s} signals={int(sig.sum()):6d} trades={len(trades):5d}")
    return pd.DataFrame(all_trades)

def equity_from_trades(trades, oos_floor=None):
    t = trades.sort_values('entry_dt').reset_index(drop=True)
    if oos_floor is not None:
        t = t[t['entry_dt'] >= oos_floor]
    if t.empty:
        return None, None
    eq = (1.0 + RISK_PCT * t['pnl_r']).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    wins = (t['pnl_r'] > 0).sum()
    losses = (t['pnl_r'] < 0).sum()
    breakeven = (t['pnl_r'] == 0).sum()
    gp = t.loc[t['pnl_r'] > 0, 'pnl_r'].sum()
    gl = -t.loc[t['pnl_r'] < 0, 'pnl_r'].sum()
    stats = dict(
        trades=len(t), wins=wins, losses=losses, breakeven=breakeven,
        win_rate=wins / len(t), profit_factor=(gp / gl if gl else np.inf),
        avg_r=t['pnl_r'].mean(), expectancy=t['pnl_r'].sum() / len(t),
        total_r=t['pnl_r'].sum(), final_eq=eq.iloc[-1],
        max_dd=dd.min(),
    )
    return pd.DataFrame({'ts': t['entry_dt'].values, 'equity': eq.values, 'dd': dd.values}), stats

def plot_equity(df_eq, stats, title, outpath):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(df_eq['ts'], df_eq['equity'], lw=1.2, color='#1f77b4')
    ax[0].axhline(1.0, color='gray', lw=0.8, ls='--')
    ax[0].set_yscale('log')
    ax[0].set_title(title)
    ax[0].grid(alpha=0.3)
    ax[1].fill_between(df_eq['ts'], df_eq['dd'], 0, color='red', alpha=0.4)
    ax[1].set_title('Drawdown (log-equity scale)')
    ax[1].grid(alpha=0.3)
    txt = (f"Trades: {stats['trades']}   Win%: {stats['win_rate']:.1%}   "
           f"PF: {stats['profit_factor']:.2f}   Avg R: {stats['avg_r']:.3f}   "
           f"Total R: {stats['total_r']:.1f}   MaxDD: {stats['max_dd']:.1%}   "
           f"Final: {stats['final_eq']:.2f}x")
    fig.suptitle(title + "\n" + txt, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)

def main():
    model = joblib.load(os.path.join(os.path.dirname(__file__), "models", "larry_model.pkl"))
    print("Reading and building features...")
    trades = run_backtest(model)
    print(f"\nTotal trades (all TF, in-sample): {len(trades)}")
    print("\n--- By symbol ---")
    g = trades.groupby('symbol').agg(trades=('pnl_r', 'size'),
                                     total_r=('pnl_r', 'sum'),
                                     win_rate=('pnl_r', lambda x: (x > 0).mean()))
    print(g.round(3).to_string())
    print("\n--- By timeframe ---")
    g = trades.groupby('tf').agg(trades=('pnl_r', 'size'),
                                 total_r=('pnl_r', 'sum'),
                                 win_rate=('pnl_r', lambda x: (x > 0).mean()))
    print(g.round(3).to_string())
    print("\n--- Exit reasons ---")
    print(trades['exit_reason'].value_counts())

    oos_floor = pd.Timestamp('2023-01-01')  # последние ~3 года как "не виденные" выборка

    df_eq, stats = equity_from_trades(trades)
    plot_equity(df_eq, stats, "Larry Williams XGBoost - ALL mid (2010-2026, in-sample)", "equity_all.png")
    print("\n--- In-sample stats (2010-2026) ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    df_eq_oos, stats_oos = equity_from_trades(trades, oos_floor)
    if df_eq_oos is not None:
        plot_equity(df_eq_oos, stats_oos, "Larry Williams XGBoost - OOS-window 2023-2026", "equity_oos.png")
        print(f"\n--- Out-of-sample-ish stats (>= {oos_floor.date()}) ---")
        for k, v in stats_oos.items():
            print(f"  {k}: {v}")

    # учитывая что сделок куча на H1, построим эквити отдельно по каждому TF
    for tf in trades['tf'].unique():
        sub = trades[trades['tf'] == tf]
        eq, st = equity_from_trades(sub)
        if eq is not None:
            plot_equity(eq, st, f"Larry Williams XGBoost - {tf} (in-sample)", f"equity_{tf}.png")
    print("\nPlots saved: equity_all.png, equity_oos.png, equity_H1/H4/D1.png")

if __name__ == "__main__":
    main()