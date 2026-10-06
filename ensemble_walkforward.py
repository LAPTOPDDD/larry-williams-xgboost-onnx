import glob, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

RR = 1.5
HORIZON = 20
TRAIN_CUT = '2021-01-01'   # обучение только до этой даты
THR = 0.6                  # порог как в EA
RISK_PER_MODEL = 0.001     # 0.1% риска на сделку каждой модели (10 моделей = до 1%)
RISK_SINGLE = 0.01         # для одиночных моделей - 1%

FEATURES = [
    "BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR",
    "RELATIVE_ATR", "DAY", "HOUR", "VOL_CHANGE", "PREV_DIR"
]

PARAM_SETS = [
    dict(lr=0.05, depth=6, n=500,  sub=0.8,  col=0.8,  seed=42,  extra={}),           # исходные
    dict(lr=0.03, depth=5, n=800,  sub=0.9,  col=0.7,  seed=7,   extra={'min_child_weight': 3}),
    dict(lr=0.08, depth=7, n=400,  sub=0.7,  col=0.9,  seed=13,  extra={}),
    dict(lr=0.02, depth=4, n=1200, sub=0.8,  col=0.6,  seed=21,  extra={'gamma': 0.5}),
    dict(lr=0.05, depth=6, n=600,  sub=0.6,  col=0.6,  seed=99,  extra={'reg_lambda': 5, 'reg_alpha': 1}),
    dict(lr=0.10, depth=3, n=500,  sub=1.0,  col=1.0,  seed=2020, extra={'min_child_weight': 5}),
    dict(lr=0.03, depth=8, n=700,  sub=0.7,  col=0.8,  seed=31,  extra={'gamma': 1.0}),
    dict(lr=0.07, depth=5, n=600,  sub=0.85, col=0.75, seed=55,  extra={'reg_alpha': 0.5}),
    dict(lr=0.05, depth=4, n=1000, sub=0.9,  col=0.7,  seed=77,  extra={'min_child_weight': 2}),
    dict(lr=0.04, depth=7, n=500,  sub=0.75, col=0.65, seed=123, extra={'gamma': 0.3}),
]

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

def simulate_symbol(df, proba, thr, start_dt):
    """Сделки по правилам EA (одна позиция за раз) только для входов >= start_dt."""
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    dt = pd.to_datetime(df['DATETIME']).values
    sig = ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values
    sig_idx = np.where(sig)[0]
    trades, open_until = [], -1
    for k, i in enumerate(sig_idx):
        if i + 1 >= len(df) or i <= open_until or dt[i + 1] < start_dt:
            continue
        pb, ps = proba[k, 1], proba[k, 2]
        if pb > thr: side = 1
        elif ps > thr: side = 2
        else: continue
        entry = o[i + 1]
        if side == 1:
            sl, r = l[i], entry - l[i]
            tp = entry + r * RR
        else:
            sl = h[i]
            if sl <= entry:
                continue
            r = sl - entry
            tp = entry - r * RR
        if r <= 0: continue
        ex, j = None, i + 1
        for j in range(i + 1, len(df)):
            if side == 1:
                if l[j] <= sl: ex = sl; break
                if h[j] >= tp: ex = tp; break
            else:
                if h[j] >= sl: ex = sl; break
                if l[j] <= tp: ex = tp; break
        if ex is None: continue
        pnl = (ex - entry) / r if side == 1 else (entry - ex) / r
        open_until = j
        trades.append((pd.Timestamp(dt[i + 1]), pnl))
    return trades

def load_universe():
    frames, metas = [], []
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.csv")))
    for f in files:
        name = os.path.basename(f).replace(".csv", "")
        symbol, tf = name.split("_")[0], name.split("_")[1]
        df = pd.read_csv(f, sep='\t')
        df = build_larry_features(df)
        df = label_larry_strategy(df)
        df = df.dropna(subset=FEATURES + ['target']).reset_index(drop=True)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        frames.append(df); metas.append((symbol, tf))
    return frames, metas

def main():
    print("Loading universe...")
    frames, metas = load_universe()

    # обучающая выборка: сигналы до cut
    sig = lambda df: ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values
    X_tr, y_tr = [], []
    for df in frames:
        m = sig(df)
        X_tr.append(df.loc[m, FEATURES].values)
        y_tr.append(df.loc[m, 'target'].values)
    X_tr = np.vstack(X_tr); y_tr = np.concatenate(y_tr)
    print(f"Train rows (signals < {TRAIN_CUT}): {len(X_tr)}")

    results = []
    for n, p in enumerate(PARAM_SETS, 1):
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            learning_rate=p['lr'], max_depth=p['depth'], n_estimators=p['n'],
            subsample=p['sub'], colsample_bytree=p['col'],
            tree_method="hist", random_state=p['seed'], **p['extra'])
        w = compute_sample_weight(class_weight='balanced', y=y_tr)
        model.fit(X_tr, y_tr, sample_weight=w)
        print(f"Model {n:2d}/{len(PARAM_SETS)} trained "
              f"(lr={p['lr']},d={p['depth']},n={p['n']},sub={p['sub']},col={p['col']},seed={p['seed']})")

        trades = []
        for df, meta in zip(frames, metas):
            m = sig(df)
            idx = np.where(m)[0]
            proba = model.predict_proba(df.loc[m, FEATURES].values)
            trades.extend(simulate_symbol(df, proba, THR, np.datetime64(TRAIN_CUT)))
        results.append((n, trades))

        t = pd.DataFrame(trades, columns=['ts', 'pnl'])
        if len(t):
            eq = (1 + RISK_SINGLE * t['pnl']).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            wr = (t['pnl'] > 0).mean()
            pf = t.loc[t['pnl'] > 0, 'pnl'].sum() / -t.loc[t['pnl'] < 0, 'pnl'].sum()
            print(f"  -> OOS trades={len(t):5d} win%={wr:.1%} PF={pf:.2f} totR={t['pnl'].sum():7.1f} "
                  f"final(1%)={eq.iloc[-1]:5.2f}x maxDD={dd:.1%}")
        else:
            print("  -> OOS: 0 trades")

    # СУММАРНАЯ эквити: одна позиция по 0.1% на сделку от каждой модели в общем счёте
    merged = []
    for n, trades in results:
        for ts, pnl in trades:
            merged.append((ts, pnl, n))
    merged = pd.DataFrame(merged, columns=['ts', 'pnl', 'model']).sort_values('ts')
    eq = (1 + RISK_PER_MODEL * merged['pnl']).cumprod()
    dd = eq / eq.cummax() - 1
    wr = (merged['pnl'] > 0).mean()
    pf = merged.loc[merged['pnl'] > 0, 'pnl'].sum() / -merged.loc[merged['pnl'] < 0, 'pnl'].sum()
    print("\n=== ENSEMBLE (10 моделей x 0.1%) ===")
    print(f"trades={len(merged)} win%={wr:.1%} PF={pf:.2f} totR={merged['pnl'].sum():.1f} "
          f"final={eq.iloc[-1]:.2f}x maxDD={dd.min():.1%}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    cmap = plt.cm.tab10
    for n, trades in results:
        if not trades: continue
        t = pd.DataFrame(trades, columns=['ts', 'pnl']).sort_values('ts')
        e = (1 + RISK_SINGLE * t['pnl']).cumprod()
        ax[0].plot(t['ts'], e, lw=0.8, alpha=0.35, color=cmap((n - 1) % 10), label=f'M{n}')
    ax[0].plot(merged['ts'], eq, lw=1.8, color='black', label='ENSEMBLE (sum)')
    ax[0].axhline(1, color='gray', lw=0.7, ls='--')
    ax[0].set_yscale('log'); ax[0].legend(fontsize=7, ncol=5); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"OOS {TRAIN_CUT}->2026 | 10 моделей + сумма  |  thr={THR}")
    ax[1].fill_between(merged['ts'], dd, 0, color='red', alpha=0.4); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"DD ансамбля: {dd.min():.1%}   final={eq.iloc[-1]:.2f}x   PF={pf:.2f}   "
                    f"win%={wr:.1%}   trades={len(merged)}")
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "equity_ensemble_oos.png"), dpi=130)
    print("Plot: equity_ensemble_oos.png")

if __name__ == "__main__":
    main()