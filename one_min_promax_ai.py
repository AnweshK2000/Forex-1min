import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# === CONFIG ===
API_KEYS = [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    'd162b35754ca4c54a13ebe7abecab4e0',
    'a7266b2503fd497496d47527a7e63b5d',
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    '09c09d58ed5e4cf4afd9a9cac8e09b5d',
    'df00920c02c54a59a426948a47095543'
]
INTERVAL = '1min'
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'AUD/CAD', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP']
MULTIPLIER = 100
api_index = 0

def get_next_api_key():
    global api_index
    key = API_KEYS[api_index % len(API_KEYS)]
    api_index += 1
    return key

def fetch_data(symbol):
    try:
        api_key = get_next_api_key()
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=300&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime')
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return pd.DataFrame()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(df):
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd - signal

def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0)
    minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0)
    tr = np.maximum.reduce([high - low, abs(high - close.shift()), abs(low - close.shift())])
    atr = pd.Series(tr).rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / (atr + 1e-6)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / (atr + 1e-6)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)) * 100
    return pd.Series(dx).rolling(window=period).mean()

def add_features(df):
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ema10'] = df['close'].ewm(span=10).mean()
    df['rsi14'] = compute_rsi(df['close'])
    df['momentum'] = df['close'] - df['close'].shift(4)
    df['macd'] = compute_macd(df)
    df['adx'] = compute_adx(df)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['volatility'] = df['high'] - df['low']
    df['open_close'] = df['open'] - df['close']
    df['high_low'] = df['high'] - df['low']
    df['close_shift1'] = df['close'].shift(1)
    df['close_shift2'] = df['close'].shift(2)
    df['return'] = df['close'].pct_change().shift(-1)
    df['target'] = np.where(df['return'] > 0.0002, 1, 0)  # Threshold for binary candle prediction
    return df.dropna()

def train_ensemble(df):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx',
                'bb_upper', 'bb_lower', 'volatility', 'open_close', 'high_low',
                'close_shift1', 'close_shift2']
    df_1 = df[df['target'] == 1]
    df_0 = df[df['target'] == 0]
    min_len = min(len(df_1), len(df_0))
    df_bal = pd.concat([
        resample(df_1, n_samples=min_len, replace=True, random_state=42),
        resample(df_0, n_samples=min_len, replace=True, random_state=42)
    ]).sample(frac=1).reset_index(drop=True)

    X = df_bal[features]
    y = df_bal['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(n_estimators=70, max_depth=3, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=70, depth=3, learning_rate=0.05, verbose=0)

    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('cat', cat)], voting='soft')

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        ensemble.fit(X_train, y_train)

    ensemble.fit(X_scaled, y)
    return ensemble, scaler

def predict(df, model, scaler, symbol):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx',
                'bb_upper', 'bb_lower', 'volatility', 'open_close', 'high_low',
                'close_shift1', 'close_shift2']
    X_pred = df[features].iloc[[-2]]
    X_scaled = scaler.transform(X_pred)
    proba = model.predict_proba(X_scaled)[0]
    signal = "BUY 📈" if proba[1] > 0.5 else "SELL 🔉"

    last = df.iloc[-2]
    confidence = sum([
        last['ema10'] > last['ma10'],
        last['momentum'] > 0,
        last['macd'] > 0,
        last['adx'] > 20,
        last['close'] < last['bb_lower'] if signal == "BUY 📈" else last['close'] > last['bb_upper']
    ])
    conf_label = "✅ Strong" if confidence >= 4 else "⚠️ Weak"
    price = round(last['close'], 5)

    return {
        "Symbol": symbol,
        "Signal": signal,
        "Prob BUY": round(proba[1], 2),
        "RSI": round(last['rsi14'], 1),
        "Confidence": conf_label,
        "Price x100": round(price * MULTIPLIER, 2)
    }

def run_signal_engine():
    results = []
    for symbol in SYMBOLS:
        print(f"📊 Analyzing {symbol}...")
        df = fetch_data(symbol)
        if df.empty or len(df) < 60:
            continue
        df = add_features(df)
        model, scaler = train_ensemble(df)
        result = predict(df, model, scaler, symbol)
        results.append(result)
    return pd.DataFrame(results)

if __name__ == "__main__":
    signals = run_signal_engine()
    print(signals.to_markdown(index=False) if not signals.empty else "❌ No signals generated.")
