import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# =====================
# CONFIG
# =====================
SYMBOLS = [
    "EURUSD","GBPUSD","USDJPY",
    "AUDUSD","XAUUSD","GBPJPY",
    "USDCAD","EURJPY","EURGBP",
    "USDCHF","NZDUSD"
]

TIMEFRAMES = {
    # "M5": mt5.TIMEFRAME_M5,
    # "M15": mt5.TIMEFRAME_M15,
    # "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

START = datetime(2010, 1, 1)
END   = datetime(2026, 1, 20)

SAVE_DIR = "data"
# =====================

mt5.initialize()
print("MT5 connected:", mt5.version())

for symbol in SYMBOLS:
    mt5.symbol_select(symbol, True)

    for tf_name, tf in TIMEFRAMES.items():
        print(f"Download {symbol} {tf_name}")

        rates = mt5.copy_rates_range(symbol, tf, START, END)
        if rates is None or len(rates) == 0:
            print("No data")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        df.rename(columns={
            "time": "DATETIME",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "tick_volume": "VOLUME"
        }, inplace=True)

        fname = f"{SAVE_DIR}/{symbol}_{tf_name}_{START.strftime('%Y%m%d')}_{END.strftime('%Y%m%d')}.csv"
        df.to_csv(fname, sep="\t", index=False)

mt5.shutdown()
print("DONE")
