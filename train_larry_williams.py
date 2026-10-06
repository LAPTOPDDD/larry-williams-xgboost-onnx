import glob, os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

RR_RATIO = 1.5
HORIZON = 20

def build_larry_features(df):
    # Clear column names from spaces and change them to uppercase.
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Debug: Print columns to verify (optional)
    # print("Columns detected:", df.columns.tolist())

    # Larry Williams Logic (Outside Bar)
    # Use capitalized column names according to the results strip()
    df['is_outside_bull'] = ((df['LOW'] < df['LOW'].shift(1)) & 
                             (df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['CLOSE'] > df['HIGH'].shift(1))).astype(int)

    df['is_outside_bear'] = ((df['HIGH'] > df['HIGH'].shift(1)) & 
                             (df['LOW'] < df['LOW'].shift(1)) & 
                             (df['CLOSE'] < df['LOW'].shift(1))).astype(int)

    # Supporting Features
    tr = np.maximum(df["HIGH"] - df["LOW"], 
                    np.maximum(abs(df["HIGH"] - df["CLOSE"].shift(1)), 
                               abs(df["LOW"] - df["CLOSE"].shift(1))))
    df["ATR"] = tr.rolling(14).mean()
    
    df["BODY_SIZE"] = abs(df["CLOSE"] - df["OPEN"]) / (df["HIGH"] - df["LOW"] + 1e-9)
    # Relative range of the last 10 candles
    df["REL_RANGE"] = (df["HIGH"] - df["LOW"]) / (df["HIGH"].rolling(10).max() - df["LOW"].rolling(10).min() + 1e-9)
    # Add a day feature (0=Monday, 4=Friday)
    # Larry Williams' pattern often differs in performance at the beginning/end of the week.
    df['DAY'] = pd.to_datetime(df['DATETIME']).dt.dayofweek

    # Add a relative volatility feature (is the signal candle much larger than average?)
    df['RELATIVE_ATR'] = (df['HIGH'] - df['LOW']) / df['ATR']

    # Add a trend feature (are the previous 3 candles all up or all down?)
    df['PREV_DIR'] = np.where(df['CLOSE'].shift(1) > df['OPEN'].shift(1), 1, -1)
    df['VOL_CHANGE'] = df['VOLUME'].pct_change() # Does volume increase during an Outside Bar?
    df['HOUR'] = pd.to_datetime(df['DATETIME']).dt.hour # If data < Daily
    
    return df

def label_larry_strategy(df):
    df['target'] = 0
    # Make sure to take the values ​​in capital letters
    h, l, c, o = df['HIGH'].values, df['LOW'].values, df['CLOSE'].values, df['OPEN'].values
    bull, bear = df['is_outside_bull'].values, df['is_outside_bear'].values
    
    for i in range(len(df) - HORIZON):
        entry_price = o[i+1] # Entry at the next open bar
        
        if bull[i] == 1:
            sl = l[i]
            tp = entry_price + (entry_price - sl) * RR_RATIO
            for j in range(i+1, i+HORIZON):
                if l[j] <= sl: break 
                if h[j] >= tp:
                    df.at[df.index[i], 'target'] = 1
                    break
                    
        elif bear[i] == 1:
            sl = h[i]
            tp = entry_price - (sl - entry_price) * RR_RATIO
            for j in range(i+1, i+HORIZON):
                if h[j] >= sl: break
                if l[j] <= tp:
                    df.at[df.index[i], 'target'] = 2
                    break
    return df

def train_model():
    all_files = glob.glob("data/*.csv")
    if not all_files:
        print("Error: Folder 'data' kosong atau tidak ditemukan!")
        return

    data_list = []
    for f in all_files:
        print(f"Reading: {os.path.basename(f)}")
        # sep=None with engine='python' will automatically detect Tab or Comma
        df = pd.read_csv(f, sep=None, engine='python')
        
        try:
            df = build_larry_features(df)
            df = label_larry_strategy(df)
            data_list.append(df)
        except KeyError as e:
            print(f"Skip file {f} due to column error: {e}")
            continue

    if not data_list: return
    
    full_df = pd.concat(data_list).dropna()
    
    # Filter: Only train data that has Larry Williams signal
    train_df = full_df[(full_df['is_outside_bull'] == 1) | (full_df['is_outside_bear'] == 1)].copy()
    
    features = [
        "BODY_SIZE",
        "REL_RANGE",
        "is_outside_bull",
        "is_outside_bear",
        "ATR",
        "RELATIVE_ATR",
        "DAY",
        "HOUR",
        "VOL_CHANGE",
        "PREV_DIR"
    ]
    
    train_df = full_df[(full_df['is_outside_bull'] == 1) | (full_df['is_outside_bear'] == 1)].copy()
    train_df = train_df.dropna(subset=features + ['target'])

    X = train_df[features]
    y = train_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)    

   
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective="multi:softprob", 
        num_class=3, 
        learning_rate=0.05, # Increase it a little so it's not too slow
        max_depth=6,        # MUCH SAFER. Prevents overfitting
        n_estimators=500,   # Just 300 to catch the main pattern
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist", 
        random_state=42
    )

    model.fit(X_train, y_train,sample_weight=weights)

    print("\n--- PERFORMANCE REPORT ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "models/larry_model.pkl")
    print("Model Saved!")

if __name__ == "__main__":
    train_model()