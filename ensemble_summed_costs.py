import glob, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

RR = 1.5
HORIZON = 20
TRAIN_CUT = '2021-01-01'
THR = 0.6
LOT = 0.01                      # фикс минимальный лот
COMMISSION_USD = 0.07           # комиссия $7 за лот round-trip -> $0.07 на 0.01 лота

FEATURES = ["BODY_SIZE", "REL_RANGE", "is_outside_bull", "is_outside_bear", "ATR",
            "RELATIVE_ATR", "DAY", "HOUR", "VOL_CHANGE", "PREV_DIR"]

PARAM_SETS = [
    dict(lr=0.05, depth=6, n=500,  sub=0.8,  col=0.8,  seed=42,  extra={}),
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

JPY = {'USDJPY', 'EURJPY', 'GBPJPY'}
DIRECT = {'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'}

def point_size(sym):
    if sym == 'XAUUSD': return 0.01
    if sym in JPY: return 0.001
    return 0.00001

def usd_per_price(sym, px, gbpusd, usdjpy):
    """USD P&L на 1 единицу изменения цены при лоте 0.01"""
    if sym == 'XAUUSD': return 1.0            # 0.01 лота = 1 унция золота
    if sym in DIRECT:   return 1000.0         # 0.01 лота * 100000 ед. валюты
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

def simulate_symbol(df, proba, thr, start_dt, gbpusd, usdjpy, costs_on):
    sym = df.attrs['symbol']
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    spread = df['SPREAD'].values
    dt = pd.to_datetime(df['DATETIME']).values
    sig = ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values
    sig_idx = np.where(sig)[0]
    pt = point_size(sym)
    trades, open_until = [], -1
    for k, i in enumerate(sig_idx):
        if i + 1 >= len(df) or i <= open_until or dt[i + 1] < start_dt:
            continue
        pb, ps = proba[k, 1], proba[k, 2]
        if pb > thr: side = 1
        elif ps > thr: side = 2
        else: continue
        eb = i + 1
        entry = o[eb]
        if side == 1:
            sl, r = l[i], entry - l[i]
            tp = entry + r * RR
        else:
            sl = h[i]
            if sl <= entry: continue
            r = sl - entry
            tp = entry - r * RR
        if r <= 0: continue
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
        gross_px = (ex - entry) if side == 1 else (entry - ex)
        upp = usd_per_price(sym, entry, gbpusd, usdjpy)
        pnl = gross_px * upp
        cost_spread = (spread[eb] * pt * upp) if costs_on else 0.0
        cost_comm = COMMISSION_USD if costs_on else 0.0
        trades.append(dict(entry=pd.Timestamp(dt[eb]), exit=pd.Timestamp(dt[j]),
                           symbol=sym, gross=pnl, cost=cost_spread + cost_comm,
                           net=pnl - cost_spread - cost_comm))
    return trades

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

def run_ensemble(frames, costs_on):
    sig = lambda df: ((df['is_outside_bull'] == 1) | (df['is_outside_bear'] == 1)).values
    X_tr, y_tr = [], []
    for df in frames:
        m = sig(df)
        X_tr.append(df.loc[m, FEATURES].values)
        y_tr.append(df.loc[m, 'target'].values)
    X_tr = np.vstack(X_tr); y_tr = np.concatenate(y_tr)

    # курсы для кроссов (последнее известное значение — для конвертации пип-валью)
    gbpusd = float(frames[[f.attrs['symbol'] for f in frames].index('GBPUSD')]['CLOSE'].iloc[-1])
    usdjpy = float(frames[[f.attrs['symbol'] for f in frames].index('USDJPY')]['CLOSE'].iloc[-1])

    per_model, all_trades = [], []
    for n, p in enumerate(PARAM_SETS, 1):
        model = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3,
            learning_rate=p['lr'], max_depth=p['depth'], n_estimators=p['n'],
            subsample=p['sub'], colsample_bytree=p['col'],
            tree_method="hist", random_state=p['seed'], **p['extra'])
        w = compute_sample_weight(class_weight='balanced', y=y_tr)
        model.fit(X_tr, y_tr, sample_weight=w)
        mt = []
        for df in frames:
            m = sig(df)
            if not m.any(): continue
            proba = model.predict_proba(df.loc[m, FEATURES].values)
            mt.extend(simulate_symbol(df, proba, THR, np.datetime64(TRAIN_CUT),
                                      gbpusd, usdjpy, costs_on))
        per_model.append(mt)
        all_trades.extend([{**t, 'model': n} for t in mt])
        net = sum(t['net'] for t in mt)
        cost = sum(t['cost'] for t in mt)
        print(f"  M{n:2d}: trades={len(mt):5d} gross=${sum(t['gross'] for t in mt):9.2f} "
              f"costs=${cost:8.2f} net=${net:9.2f}")
    return per_model, pd.DataFrame(all_trades)

def equity_curve(df, col='net'):
    df = df.sort_values('exit').reset_index(drop=True)
    return df['exit'].values, df[col].cumsum().values

def main():
    print(f"ОТСТРОЙКА И ВАЛИДАЦИЯ: обучение < {TRAIN_CUT}, тест {TRAIN_CUT}->2026")
    print(f"Лот={LOT} фикс, порог={THR}, комиссия ${COMMISSION_USD}/0.01 лот round-trip, "
          f"спред из CSV\n")
    print("Загрузка данных...")
    frames = load_universe()

    print("\n--- Обучение и тест (С издержками) ---")
    per_model_cost, T = run_ensemble(frames, costs_on=True)
    print("\n--- Тест (БЕЗ издержек, для сравнения) ---")
    per_model_gross, Tg = run_ensemble(frames, costs_on=False)

    def stats(d):
        n = len(d); gp = d.loc[d['net'] > 0, 'net'].sum(); gl = -d.loc[d['net'] < 0, 'net'].sum()
        eq = d.sort_values('exit')['net'].cumsum()
        dd = eq.cummax() - eq
        return dict(trades=n, win=(d['net'] > 0).mean(), pf=gp / gl if gl else np.inf,
                    net=d['net'].sum(), gross=d['gross'].sum(), cost=d['cost'].sum(),
                    maxdd=dd.max())

    s, sg = stats(T), stats(Tg)
    print("\n=== СУММАРНАЯ КОРЗИНА (10 моделей, деньги сложены, 0.01 лот) ===")
    print(f"{'':16s}{'С издержками':>16s}{'Без издержек':>16s}")
    print(f"{'сделок':16s}{s['trades']:16d}{sg['trades']:16d}")
    print(f"{'win%':16s}{s['win']:16.1%}{sg['win']:16.1%}")
    print(f"{'gross $':16s}{s['gross']:16.2f}{sg['gross']:16.2f}")
    print(f"{'издержки $':16s}{s['cost']:16.2f}{0:16.2f}")
    print(f"{'PF (net)':16s}{s['pf']:16.2f}{sg['pf']:16.2f}")
    print(f"{'NET $':16s}{s['net']:16.2f}{sg['net']:16.2f}")
    print(f"{'MaxDD $':16s}{s['maxdd']:16.2f}{sg['maxdd']:16.2f}")
    print(f"\nИздержки съели {s['cost'] / s['gross']:.1%} gross P&L")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # сумма отдельных эквити (ровно та же корзина) + вклад каждой модели
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for n, mt in enumerate(per_model_cost, 1):
        d = pd.DataFrame(mt)
        if d.empty: continue
        x, y = equity_curve(d)
        ax[0].step(x, y, where='post', lw=0.8, alpha=0.3, label=f'M{n}')
    x, y = equity_curve(T)
    xg, yg = equity_curve(Tg)
    ax[0].step(x, y, where='post', lw=2.2, color='black', label='СУММА с издержками')
    ax[0].step(xg, yg, where='post', lw=1.2, color='gray', ls='--', label='СУММА без издержек')
    ax[0].axhline(0, color='gray', lw=0.7)
    ax[0].set_ylabel('P&L, $ (фикс 0.01 лот)'); ax[0].legend(fontsize=7, ncol=4); ax[0].grid(alpha=0.3)
    dd = pd.Series(y).cummax() - pd.Series(y)
    ax[1].step(x, dd, where='post', color='red', lw=1.5); ax[1].grid(alpha=0.3)
    ax[1].set_ylabel('MaxDD, $')
    fig.suptitle(f"Корзина 10 моделей, лот 0.01 фикс, OOS {TRAIN_CUT}->2026 | "
                 f"net=${s['net']:.2f} | PF={s['pf']:.2f} | издержки=${s['cost']:.2f} "
                 f"({s['cost']/s['gross']:.0%} gross) | MaxDD=${s['maxdd']:.2f} | сделок={s['trades']}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(os.path.dirname(__file__), "equity_summed_costs.png"), dpi=130)
    print("Plot: equity_summed_costs.png")

if __name__ == '__main__':
    main()