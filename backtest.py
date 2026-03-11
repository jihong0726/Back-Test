import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

VERSION = "V8.3_STABLE_FIX2"

CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"],
    "interval": "5m",
    "bars": 3000,
    "fee": 0.0006,
    "slippage": 0.0002,
}

REPORT_DIR = "reports"


def ensure_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------------------------------
# 数据抓取
# -------------------------------------------------
def normalize_candle_row(row):
    """
    Bitget candle 常见格式：
    [ts, open, high, low, close, baseVol, quoteVol]
    有时可能有更多字段，这里只取前 7 个
    """
    if not isinstance(row, (list, tuple)) or len(row) < 6:
        return None

    row = list(row)

    if len(row) >= 7:
        return row[:7]

    # 若只有 6 个字段，补一个 quote volume 占位
    return row[:6] + [np.nan]


def fetch_bitget(symbol, limit=3000):
    url = "https://api.bitget.com/api/v2/mix/market/candles"

    rows = []
    end_time = int(time.time() * 1000)

    while len(rows) < limit:
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": "5m",
            "limit": "1000",
            "endTime": str(end_time),
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()

        payload = r.json()
        data = payload.get("data", [])

        if not data:
            break

        batch = []
        for item in data:
            norm = normalize_candle_row(item)
            if norm is not None:
                batch.append(norm)

        if not batch:
            break

        rows.extend(batch)

        oldest = min(int(float(x[0])) for x in batch)
        end_time = oldest - 1

        time.sleep(0.1)

        if len(batch) < 2:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    )

    # 关键修复：先转 numeric，再转 datetime
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 再转 datetime，避免字符串被直接解析导致溢出
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    return df.tail(limit).reset_index(drop=True)


# -------------------------------------------------
# 指标
# -------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def calc_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        )
    )

    atr = tr.rolling(period, min_periods=period).mean()
    return atr.fillna(method="bfill") if hasattr(atr, "fillna") else atr


def calc_vwap(df):
    price = (df["high"] + df["low"] + df["close"]) / 3
    pv = price * df["volume"]

    cum_pv = pv.cumsum()
    cum_vol = df["volume"].replace(0, np.nan).cumsum()

    vwap = cum_pv / cum_vol
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill()

    return vwap


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["rsi"] = calc_rsi(df["close"], 7)
    df["atr"] = calc_atr(df, 14)
    df["vwap"] = calc_vwap(df)

    df["dist_ema20"] = df["close"] / df["ema20"] - 1
    df["dist_vwap"] = df["close"] / df["vwap"] - 1

    return df


# -------------------------------------------------
# 策略
# -------------------------------------------------
def generate_strategies(df):
    s = {}

    s["RSI_20_80"] = np.where(
        df["rsi"] < 20, 1,
        np.where(df["rsi"] > 80, -1, 0)
    )

    s["RSI_25_75"] = np.where(
        df["rsi"] < 25, 1,
        np.where(df["rsi"] > 75, -1, 0)
    )

    s["EMA20_dev_1.5"] = np.where(
        df["dist_ema20"] < -0.015, 1,
        np.where(df["dist_ema20"] > 0.015, -1, 0)
    )

    s["EMA20_dev_2.0"] = np.where(
        df["dist_ema20"] < -0.02, 1,
        np.where(df["dist_ema20"] > 0.02, -1, 0)
    )

    s["VWAP_1.5"] = np.where(
        df["dist_vwap"] < -0.015, 1,
        np.where(df["dist_vwap"] > 0.015, -1, 0)
    )

    s["VWAP_2.0"] = np.where(
        df["dist_vwap"] < -0.02, 1,
        np.where(df["dist_vwap"] > 0.02, -1, 0)
    )

    return s


# -------------------------------------------------
# 回测
# -------------------------------------------------
def backtest(df, signal, fee=0.0006, slippage=0.0002):
    signal = pd.Series(signal, index=df.index).fillna(0)

    # 信号延续持仓
    position = signal.replace(0, np.nan).ffill().fillna(0)
    position = position.shift(1).fillna(0)

    ret = df["close"].pct_change().fillna(0)
    gross = position * ret

    trades = position.diff().abs().fillna(0)
    cost = trades * (fee + slippage)

    net = gross - cost

    equity = (1 + net).cumprod()
    total_return = (equity.iloc[-1] - 1) * 100

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min() * 100

    sharpe = 0.0
    if net.std() not in [0, np.nan] and pd.notna(net.std()) and net.std() > 0:
        sharpe = net.mean() / net.std() * np.sqrt(252 * 24 * 12)

    return total_return, max_dd, sharpe


# -------------------------------------------------
# 主程序
# -------------------------------------------------
def main():
    ensure_dir()

    summary = []

    for sym in CONFIG["symbols"]:
        print("processing", sym)

        df = fetch_bitget(sym, CONFIG["bars"])

        if df.empty or len(df) < 100:
            print(f"skip {sym}, insufficient data")
            continue

        df = add_indicators(df)
        strategies = generate_strategies(df)

        for name, signal in strategies.items():
            total, dd, sharpe = backtest(
                df,
                signal,
                fee=CONFIG["fee"],
                slippage=CONFIG["slippage"]
            )

            summary.append({
                "symbol": sym,
                "strategy": name,
                "return_pct": total,
                "max_dd_pct": dd,
                "sharpe": sharpe
            })

    if not summary:
        with open("latest_summary.txt", "w", encoding="utf-8") as f:
            f.write("No valid results generated.")
        print("No valid results generated.")
        return

    res = pd.DataFrame(summary)
    res = res.sort_values(["return_pct", "sharpe"], ascending=[False, False]).reset_index(drop=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{REPORT_DIR}/result_{now}.csv"

    res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open("latest_summary.txt", "w", encoding="utf-8") as f:
        f.write(res.head(20).to_string(index=False))

    print(res.head(20).to_string(index=False))
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
