import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

VERSION = "V9_RESEARCH_ENGINE"
REPORT_DIR = "reports"

CONFIG = {
    "symbols": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
        "MATICUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT",
        "ATOMUSDT", "NEARUSDT", "DOTUSDT", "FILUSDT", "INJUSDT"
    ],
    "interval": "5m",
    "bars": 20000,
    "page_limit": 1000,
    "max_pages": 25,
    "fee": 0.0006,
    "slippage": 0.0002,
    "train_ratio": 0.6,
    "min_rows": 3000
}


def ensure_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------------------------------
# Data Fetch
# -------------------------------------------------
def interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1H": 3_600_000,
        "4H": 14_400_000,
        "1D": 86_400_000,
    }
    return mapping.get(interval, 300_000)


def normalize_candle_row(row):
    if not isinstance(row, (list, tuple)) or len(row) < 6:
        return None

    row = list(row)
    if len(row) >= 7:
        return row[:7]
    return row[:6] + [np.nan]


def fetch_bitget(symbol, interval="5m", limit=20000, page_limit=1000, max_pages=25):
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    step_ms = interval_to_ms(interval)

    rows = []
    seen_ts = set()
    end_time = int(time.time() * 1000)

    print(f"[{symbol}] start fetch, target={limit}")

    for page in range(max_pages):
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "limit": str(min(page_limit, 1000)),
            "endTime": str(end_time),
        }

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()

        payload = r.json()
        data = payload.get("data", [])

        if not data:
            print(f"[{symbol}] page {page + 1}: empty")
            break

        batch = []
        for item in data:
            norm = normalize_candle_row(item)
            if norm is None:
                continue
            ts = int(float(norm[0]))
            if ts not in seen_ts:
                seen_ts.add(ts)
                batch.append(norm)

        if not batch:
            print(f"[{symbol}] page {page + 1}: duplicated")
            break

        rows.extend(batch)

        oldest = min(int(float(x[0])) for x in batch)
        newest = max(int(float(x[0])) for x in batch)

        print(f"[{symbol}] page {page + 1}: +{len(batch)} rows, {oldest} -> {newest}, total={len(rows)}")

        if len(rows) >= limit:
            break

        end_time = oldest - step_ms
        time.sleep(0.08)

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "open", "high", "low", "close", "volume", "quote_volume"
        ])

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    return df.tail(limit).reset_index(drop=True)


# -------------------------------------------------
# Indicators
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

    return tr.rolling(period, min_periods=period).mean().bfill()


def calc_vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]

    cum_pv = pv.cumsum()
    cum_vol = df["volume"].replace(0, np.nan).cumsum()

    vwap = cum_pv / cum_vol
    return vwap.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi7"] = calc_rsi(df["close"], 7)
    df["rsi14"] = calc_rsi(df["close"], 14)
    df["atr14"] = calc_atr(df, 14)
    df["vwap"] = calc_vwap(df)

    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["ema_dev"] = df["close"] / df["ema20"] - 1
    df["vwap_dev"] = df["close"] / df["vwap"] - 1

    return df


# -------------------------------------------------
# Market State
# -------------------------------------------------
def detect_market_state(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "UNKNOWN"

    last = df.iloc[-1]
    ema_spread = abs(last["ema20"] - last["ema50"]) / max(last["close"], 1e-9)
    atr_ratio = last["atr14"] / max(last["close"], 1e-9)

    if ema_spread < 0.003 and atr_ratio < 0.015:
        return "RANGE"

    if last["ema20"] > last["ema50"]:
        return "TREND_UP"

    if last["ema20"] < last["ema50"]:
        return "TREND_DOWN"

    return "NEUTRAL"


# -------------------------------------------------
# Strategies
# -------------------------------------------------
def generate_strategies(df):
    strategies = {}

    strategies["VWAP_1.5"] = np.where(
        df["vwap_dev"] < -0.015, 1,
        np.where(df["vwap_dev"] > 0.015, -1, 0)
    )

    strategies["VWAP_2.0"] = np.where(
        df["vwap_dev"] < -0.020, 1,
        np.where(df["vwap_dev"] > 0.020, -1, 0)
    )

    strategies["VWAP_2.5"] = np.where(
        df["vwap_dev"] < -0.025, 1,
        np.where(df["vwap_dev"] > 0.025, -1, 0)
    )

    strategies["EMA_DEV_1.5"] = np.where(
        df["ema_dev"] < -0.015, 1,
        np.where(df["ema_dev"] > 0.015, -1, 0)
    )

    strategies["EMA_DEV_2.0"] = np.where(
        df["ema_dev"] < -0.020, 1,
        np.where(df["ema_dev"] > 0.020, -1, 0)
    )

    strategies["RSI_20_80"] = np.where(
        df["rsi7"] < 20, 1,
        np.where(df["rsi7"] > 80, -1, 0)
    )

    strategies["RSI_25_75"] = np.where(
        df["rsi7"] < 25, 1,
        np.where(df["rsi7"] > 75, -1, 0)
    )

    strategies["RSI_30_70"] = np.where(
        df["rsi7"] < 30, 1,
        np.where(df["rsi7"] > 70, -1, 0)
    )

    return strategies


# -------------------------------------------------
# Backtest
# -------------------------------------------------
def build_position(signal):
    position = pd.Series(signal).replace(0, np.nan).ffill().fillna(0)
    position = position.shift(1).fillna(0)
    return position


def backtest_core(df, signal, fee=0.0006, slippage=0.0002):
    position = build_position(signal)
    ret = df["ret"]

    gross = position * ret
    trades = position.diff().abs().fillna(0)
    cost = trades * (fee + slippage)
    net = gross - cost

    equity = (1 + net).cumprod()
    total_return = (equity.iloc[-1] - 1) * 100

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min() * 100

    std = net.std()
    sharpe = 0.0
    if pd.notna(std) and std > 0:
        sharpe = net.mean() / std * np.sqrt(252 * 24 * 12)

    trade_count = int((trades > 0).sum())

    active = net[position != 0]
    win_rate = 0.0
    if len(active) > 0:
        win_rate = float((active > 0).mean())

    # average holding bars
    holding_lengths = []
    current_len = 0
    pos_arr = position.to_numpy()

    for i in range(len(pos_arr)):
        if pos_arr[i] != 0:
            current_len += 1
        elif current_len > 0:
            holding_lengths.append(current_len)
            current_len = 0
    if current_len > 0:
        holding_lengths.append(current_len)

    avg_holding = float(np.mean(holding_lengths)) if holding_lengths else 0.0

    return {
        "return_pct": float(total_return),
        "max_dd_pct": float(max_dd),
        "sharpe": float(sharpe),
        "trades": trade_count,
        "win_rate": float(win_rate),
        "avg_holding_bars": avg_holding,
    }


def train_test_backtest(df, signal, fee, slippage, train_ratio):
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_signal = pd.Series(signal[:split_idx], index=train_df.index)
    test_signal = pd.Series(signal[split_idx:], index=test_df.index)

    train_res = backtest_core(train_df, train_signal, fee, slippage)
    test_res = backtest_core(test_df, test_signal, fee, slippage)
    full_res = backtest_core(df, pd.Series(signal, index=df.index), fee, slippage)

    return train_res, test_res, full_res


# -------------------------------------------------
# Realtime Signals
# -------------------------------------------------
def generate_live_signal(df, symbol, market):
    last = df.iloc[-1]

    action = None
    strategy = None
    entry = float(last["close"])
    tp = None
    sl = None

    # Range markets: prioritize VWAP reversion
    if market == "RANGE":
        if last["vwap_dev"] < -0.02:
            action = "LONG"
            strategy = "VWAP_2.0"
            tp = float(last["vwap"])
            sl = float(last["close"] - last["atr14"])
        elif last["vwap_dev"] > 0.02:
            action = "SHORT"
            strategy = "VWAP_2.0"
            tp = float(last["vwap"])
            sl = float(last["close"] + last["atr14"])
        elif last["vwap_dev"] < -0.015:
            action = "LONG"
            strategy = "VWAP_1.5"
            tp = float(last["vwap"])
            sl = float(last["close"] - last["atr14"])
        elif last["vwap_dev"] > 0.015:
            action = "SHORT"
            strategy = "VWAP_1.5"
            tp = float(last["vwap"])
            sl = float(last["close"] + last["atr14"])

    # Trend markets: very light trend bias
    elif market == "TREND_UP":
        if last["close"] > last["ema20"] and last["rsi7"] < 35:
            action = "LONG"
            strategy = "TrendPullback"
            tp = float(last["close"] + last["atr14"] * 1.5)
            sl = float(last["close"] - last["atr14"])

    elif market == "TREND_DOWN":
        if last["close"] < last["ema20"] and last["rsi7"] > 65:
            action = "SHORT"
            strategy = "TrendPullback"
            tp = float(last["close"] - last["atr14"] * 1.5)
            sl = float(last["close"] + last["atr14"])

    if action is None:
        return None

    return {
        "symbol": symbol,
        "market": market,
        "strategy": strategy,
        "action": action,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "close": float(last["close"]),
        "rsi7": float(last["rsi7"]),
        "vwap_dev_pct": float(last["vwap_dev"] * 100),
        "ema_dev_pct": float(last["ema_dev"] * 100),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ensure_dir()

    all_rows = []
    best_rows = []
    signal_rows = []

    for sym in CONFIG["symbols"]:
        try:
            print("processing", sym)

            df = fetch_bitget(
                sym,
                interval=CONFIG["interval"],
                limit=CONFIG["bars"],
                page_limit=CONFIG["page_limit"],
                max_pages=CONFIG["max_pages"]
            )

            if df.empty or len(df) < CONFIG["min_rows"]:
                print(f"skip {sym}, insufficient data: {len(df)}")
                continue

            df = add_indicators(df)
            market = detect_market_state(df)
            strategy_dict = generate_strategies(df)

            symbol_rows = []

            for name, signal in strategy_dict.items():
                train_res, test_res, full_res = train_test_backtest(
                    df,
                    signal,
                    CONFIG["fee"],
                    CONFIG["slippage"],
                    CONFIG["train_ratio"]
                )

                row = {
                    "symbol": sym,
                    "market": market,
                    "strategy": name,

                    "train_return_pct": train_res["return_pct"],
                    "train_max_dd_pct": train_res["max_dd_pct"],
                    "train_sharpe": train_res["sharpe"],
                    "train_trades": train_res["trades"],
                    "train_win_rate": train_res["win_rate"],

                    "test_return_pct": test_res["return_pct"],
                    "test_max_dd_pct": test_res["max_dd_pct"],
                    "test_sharpe": test_res["sharpe"],
                    "test_trades": test_res["trades"],
                    "test_win_rate": test_res["win_rate"],

                    "full_return_pct": full_res["return_pct"],
                    "full_max_dd_pct": full_res["max_dd_pct"],
                    "full_sharpe": full_res["sharpe"],
                    "full_trades": full_res["trades"],
                    "full_win_rate": full_res["win_rate"],
                    "avg_holding_bars": full_res["avg_holding_bars"],
                }

                # score: prioritize test performance
                score = (
                    row["test_return_pct"] * 1.5
                    + row["test_sharpe"] * 8
                    - abs(row["test_max_dd_pct"]) * 1.2
                    + row["test_win_rate"] * 10
                )
                row["score"] = score

                all_rows.append(row)
                symbol_rows.append(row)

            if symbol_rows:
                best = pd.DataFrame(symbol_rows).sort_values(
                    ["score", "test_return_pct", "test_sharpe"],
                    ascending=[False, False, False]
                ).iloc[0].to_dict()
                best_rows.append(best)

            live_signal = generate_live_signal(df, sym, market)
            if live_signal is not None:
                signal_rows.append(live_signal)

        except Exception as e:
            print(f"error on {sym}: {e}")

    if not all_rows:
        with open("latest_summary.txt", "w", encoding="utf-8") as f:
            f.write("No valid results generated.")
        print("No valid results generated.")
        return

    all_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(best_rows)
    signal_df = pd.DataFrame(signal_rows)

    all_df = all_df.sort_values(
        ["score", "test_return_pct", "test_sharpe"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    best_df = best_df.sort_values(
        ["score", "test_return_pct", "test_sharpe"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    rank_path = f"{REPORT_DIR}/strategy_rank_{now}.csv"
    best_path = f"{REPORT_DIR}/best_per_symbol_{now}.csv"
    signal_path = f"{REPORT_DIR}/signals_{now}.csv"

    all_df.to_csv(rank_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append(f"Version: {VERSION}")
    summary_lines.append(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    summary_lines.append("=== TOP 20 STRATEGIES ===")
    summary_lines.append(
        all_df[[
            "symbol", "market", "strategy",
            "test_return_pct", "test_max_dd_pct", "test_sharpe",
            "test_trades", "test_win_rate", "score"
        ]].head(20).to_string(index=False)
    )
    summary_lines.append("")
    summary_lines.append("=== BEST PER SYMBOL ===")
    if not best_df.empty:
        summary_lines.append(
            best_df[[
                "symbol", "market", "strategy",
                "test_return_pct", "test_max_dd_pct", "test_sharpe",
                "test_trades", "test_win_rate", "score"
            ]].to_string(index=False)
        )
    else:
        summary_lines.append("No best rows.")

    summary_lines.append("")
    summary_lines.append("=== LIVE SIGNALS ===")
    if not signal_df.empty:
        summary_lines.append(signal_df.to_string(index=False))
    else:
        summary_lines.append("No live signals.")

    with open("latest_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print("")
    print("saved:", rank_path)
    print("saved:", best_path)
    print("saved:", signal_path)


if __name__ == "__main__":
    main()
