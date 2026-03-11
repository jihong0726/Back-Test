import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

VERSION = "V10_LIVE_ENGINE"

CONFIG = {
    "symbols": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
        "MATICUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT",
        "ATOMUSDT", "NEARUSDT", "DOTUSDT", "FILUSDT", "INJUSDT"
    ],
    "interval": "5m",
    "bars": 1500,
    "page_limit": 1000,
    "max_pages": 6,
    "max_open_positions": 5,
    "min_signal_score": 68,
    "fee": 0.0006,
    "slippage": 0.0002,
    "atr_stop_mult": 1.0,
    "atr_tp_mult": 1.8,
    "partial_tp_ratio": 0.5,
    "move_sl_after_profit_atr": 1.0,
}

STATE_DIR = "state"
REPORT_DIR = "reports"
POSITIONS_PATH = os.path.join(STATE_DIR, "positions.json")
DECISION_LOG_PATH = os.path.join(REPORT_DIR, "decision_log.csv")
SIGNALS_PATH = os.path.join(REPORT_DIR, "signals.csv")
SUMMARY_PATH = os.path.join(REPORT_DIR, "latest_summary.txt")


# =========================================================
# IO Helpers
# =========================================================
def ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def load_positions() -> dict:
    if not os.path.exists(POSITIONS_PATH):
        return {}

    with open(POSITIONS_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def save_positions(positions: dict):
    with open(POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)


def append_decision_logs(rows: list[dict]):
    if not rows:
        return

    df = pd.DataFrame(rows)
    if os.path.exists(DECISION_LOG_PATH):
        old = pd.read_csv(DECISION_LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(DECISION_LOG_PATH, index=False, encoding="utf-8-sig")


# =========================================================
# Market Data
# =========================================================
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


def fetch_bitget(symbol, interval="5m", limit=1500, page_limit=1000, max_pages=6):
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    step_ms = interval_to_ms(interval)

    rows = []
    seen_ts = set()
    end_time = int(time.time() * 1000)

    for _ in range(max_pages):
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
            break

        rows.extend(batch)

        if len(rows) >= limit:
            break

        oldest = min(int(float(x[0])) for x in batch)
        end_time = oldest - step_ms
        time.sleep(0.05)

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


# =========================================================
# Indicators
# =========================================================
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
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
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

    df["vwap_dev"] = df["close"] / df["vwap"] - 1
    df["ema_dev"] = df["close"] / df["ema20"] - 1
    df["ema_spread"] = (df["ema20"] - df["ema50"]) / df["close"]
    df["atr_ratio"] = df["atr14"] / df["close"]

    return df


# =========================================================
# Market Regime
# =========================================================
def detect_market_state(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "UNKNOWN"

    last = df.iloc[-1]
    ema_spread = abs(float(last["ema_spread"]))
    atr_ratio = float(last["atr_ratio"])

    if ema_spread < 0.003 and atr_ratio < 0.02:
        return "RANGE"

    if last["ema20"] > last["ema50"]:
        return "TREND_UP"

    if last["ema20"] < last["ema50"]:
        return "TREND_DOWN"

    return "NEUTRAL"


# =========================================================
# Signal Scoring
# =========================================================
def calc_signal_score(market, action, vwap_dev, ema_dev, rsi7):
    score = 50.0
    score += min(abs(vwap_dev) * 1500, 28)
    score += min(abs(ema_dev) * 600, 15)

    if market == "RANGE":
        score += 8
    elif market.startswith("TREND"):
        score += 4

    if action == "LONG":
        if rsi7 < 20:
            score += 10
        elif rsi7 < 30:
            score += 6
        elif rsi7 < 40:
            score += 3
    elif action == "SHORT":
        if rsi7 > 80:
            score += 10
        elif rsi7 > 70:
            score += 6
        elif rsi7 > 60:
            score += 3

    return round(min(score, 100), 2)


# =========================================================
# Entry Signal Engine
# =========================================================
def build_entry_signal(symbol, df, market):
    last = df.iloc[-1]

    close = float(last["close"])
    vwap = float(last["vwap"])
    atr = float(last["atr14"])
    rsi7 = float(last["rsi7"])
    vwap_dev = float(last["vwap_dev"])
    ema_dev = float(last["ema_dev"])

    if market == "RANGE":
        if vwap_dev <= -0.025:
            action = "LONG"
            strategy = "VWAP_2.5"
        elif vwap_dev >= 0.025:
            action = "SHORT"
            strategy = "VWAP_2.5"
        elif vwap_dev <= -0.020:
            action = "LONG"
            strategy = "VWAP_2.0"
        elif vwap_dev >= 0.020:
            action = "SHORT"
            strategy = "VWAP_2.0"
        elif vwap_dev <= -0.015:
            action = "LONG"
            strategy = "VWAP_1.5"
        elif vwap_dev >= 0.015:
            action = "SHORT"
            strategy = "VWAP_1.5"
        else:
            return None

        score = calc_signal_score(market, action, vwap_dev, ema_dev, rsi7)
        if score < CONFIG["min_signal_score"]:
            return None

        sl = close - atr * CONFIG["atr_stop_mult"] if action == "LONG" else close + atr * CONFIG["atr_stop_mult"]
        tp = vwap

        return {
            "symbol": symbol,
            "market": market,
            "strategy": strategy,
            "action": action,
            "entry": round(close, 8),
            "tp": round(float(tp), 8),
            "sl": round(float(sl), 8),
            "score": score,
            "rsi7": round(rsi7, 4),
            "vwap_dev_pct": round(vwap_dev * 100, 4),
            "ema_dev_pct": round(ema_dev * 100, 4),
            "opened_at": utc_now_str(),
        }

    if market == "TREND_UP":
        if close > float(last["ema20"]) and rsi7 < 35:
            score = calc_signal_score(market, "LONG", vwap_dev, ema_dev, rsi7)
            if score >= CONFIG["min_signal_score"]:
                return {
                    "symbol": symbol,
                    "market": market,
                    "strategy": "TrendPullback",
                    "action": "LONG",
                    "entry": round(close, 8),
                    "tp": round(close + atr * CONFIG["atr_tp_mult"], 8),
                    "sl": round(close - atr * CONFIG["atr_stop_mult"], 8),
                    "score": score,
                    "rsi7": round(rsi7, 4),
                    "vwap_dev_pct": round(vwap_dev * 100, 4),
                    "ema_dev_pct": round(ema_dev * 100, 4),
                    "opened_at": utc_now_str(),
                }

    if market == "TREND_DOWN":
        if close < float(last["ema20"]) and rsi7 > 65:
            score = calc_signal_score(market, "SHORT", vwap_dev, ema_dev, rsi7)
            if score >= CONFIG["min_signal_score"]:
                return {
                    "symbol": symbol,
                    "market": market,
                    "strategy": "TrendPullback",
                    "action": "SHORT",
                    "entry": round(close, 8),
                    "tp": round(close - atr * CONFIG["atr_tp_mult"], 8),
                    "sl": round(close + atr * CONFIG["atr_stop_mult"], 8),
                    "score": score,
                    "rsi7": round(rsi7, 4),
                    "vwap_dev_pct": round(vwap_dev * 100, 4),
                    "ema_dev_pct": round(ema_dev * 100, 4),
                    "opened_at": utc_now_str(),
                }

    return None


# =========================================================
# Position Management
# =========================================================
def current_pnl_pct(side: str, entry: float, current: float) -> float:
    if side == "LONG":
        return (current / entry - 1) * 100
    return (entry / current - 1) * 100


def update_position(symbol: str, pos: dict, df: pd.DataFrame, market: str):
    last = df.iloc[-1]
    current = float(last["close"])
    atr = float(last["atr14"])
    ema20 = float(last["ema20"])
    vwap = float(last["vwap"])

    side = pos["side"]
    entry = float(pos["entry"])
    tp = float(pos["tp"])
    sl = float(pos["sl"])
    status = pos.get("status", "OPEN")
    partial_taken = bool(pos.get("partial_taken", False))
    size = float(pos.get("size", 1.0))

    result = {
        "symbol": symbol,
        "action": "HOLD",
        "reason": "",
        "current_price": round(current, 8),
        "pnl_pct": round(current_pnl_pct(side, entry, current), 4),
        "market": market,
        "strategy": pos.get("strategy", "-"),
    }

    if status != "OPEN":
        result["action"] = "CLOSED"
        result["reason"] = "position already closed"
        return pos, result

    # Stop loss / take profit
    if side == "LONG":
        if current <= sl:
            pos["status"] = "CLOSED"
            pos["closed_at"] = utc_now_str()
            pos["exit_price"] = round(current, 8)
            pos["exit_reason"] = "STOP_LOSS"
            result["action"] = "CLOSE"
            result["reason"] = "触发止损"
            return pos, result

        if current >= tp:
            pos["status"] = "CLOSED"
            pos["closed_at"] = utc_now_str()
            pos["exit_price"] = round(current, 8)
            pos["exit_reason"] = "TAKE_PROFIT"
            result["action"] = "CLOSE"
            result["reason"] = "触发止盈"
            return pos, result

    else:
        if current >= sl:
            pos["status"] = "CLOSED"
            pos["closed_at"] = utc_now_str()
            pos["exit_price"] = round(current, 8)
            pos["exit_reason"] = "STOP_LOSS"
            result["action"] = "CLOSE"
            result["reason"] = "触发止损"
            return pos, result

        if current <= tp:
            pos["status"] = "CLOSED"
            pos["closed_at"] = utc_now_str()
            pos["exit_price"] = round(current, 8)
            pos["exit_reason"] = "TAKE_PROFIT"
            result["action"] = "CLOSE"
            result["reason"] = "触发止盈"
            return pos, result

    # Partial take profit logic
    if not partial_taken:
        if side == "LONG" and current >= entry + atr:
            pos["partial_taken"] = True
            pos["size"] = round(size * (1 - CONFIG["partial_tp_ratio"]), 4)
            pos["sl"] = round(max(sl, entry), 8)
            result["action"] = "PARTIAL_TP"
            result["reason"] = "达到第一目标，部分止盈并上移止损到入场附近"
            return pos, result

        if side == "SHORT" and current <= entry - atr:
            pos["partial_taken"] = True
            pos["size"] = round(size * (1 - CONFIG["partial_tp_ratio"]), 4)
            pos["sl"] = round(min(sl, entry), 8)
            result["action"] = "PARTIAL_TP"
            result["reason"] = "达到第一目标，部分止盈并下移止损到入场附近"
            return pos, result

    # Move stop loss if profit moves further
    move_trigger = CONFIG["move_sl_after_profit_atr"] * atr
    if side == "LONG" and current >= entry + move_trigger:
        new_sl = round(max(sl, ema20, entry), 8)
        if new_sl > sl:
            pos["sl"] = new_sl
            result["action"] = "MOVE_SL"
            result["reason"] = "盈利扩大，上移止损"
            return pos, result

    if side == "SHORT" and current <= entry - move_trigger:
        new_sl = round(min(sl, ema20, entry), 8)
        if new_sl < sl:
            pos["sl"] = new_sl
            result["action"] = "MOVE_SL"
            result["reason"] = "盈利扩大，下移止损"
            return pos, result

    # Structure invalidation
    if pos.get("strategy", "").startswith("VWAP_") and market not in ("RANGE", "NEUTRAL"):
        if side == "LONG" and current > vwap:
            result["action"] = "HOLD"
            result["reason"] = "价格仍在回归，继续持有"
            return pos, result
        if side == "SHORT" and current < vwap:
            result["action"] = "HOLD"
            result["reason"] = "价格仍在回归，继续持有"
            return pos, result

    result["action"] = "HOLD"
    result["reason"] = "结构未失效，继续持有"
    return pos, result


# =========================================================
# Summary Builders
# =========================================================
def build_summary(market_rows: list[dict], open_positions: dict, decision_rows: list[dict], new_signals: list[dict]):
    lines = []
    lines.append(f"Version: {VERSION}")
    lines.append(f"Generated At: {utc_now_str()}")
    lines.append("")

    lines.append("=== MARKET STATES ===")
    if market_rows:
        mdf = pd.DataFrame(market_rows)
        lines.append(mdf.to_string(index=False))
    else:
        lines.append("No market data.")
    lines.append("")

    lines.append("=== OPEN POSITIONS ===")
    open_rows = []
    for symbol, pos in open_positions.items():
        if pos.get("status") == "OPEN":
            open_rows.append({
                "symbol": symbol,
                "side": pos.get("side"),
                "strategy": pos.get("strategy"),
                "entry": pos.get("entry"),
                "tp": pos.get("tp"),
                "sl": pos.get("sl"),
                "size": pos.get("size"),
                "partial_taken": pos.get("partial_taken", False),
                "opened_at": pos.get("opened_at"),
            })
    if open_rows:
        lines.append(pd.DataFrame(open_rows).to_string(index=False))
    else:
        lines.append("No open positions.")
    lines.append("")

    lines.append("=== POSITION DECISIONS ===")
    if decision_rows:
        lines.append(pd.DataFrame(decision_rows).to_string(index=False))
    else:
        lines.append("No position decisions.")
    lines.append("")

    lines.append("=== NEW ENTRY SIGNALS ===")
    if new_signals:
        lines.append(pd.DataFrame(new_signals).to_string(index=False))
    else:
        lines.append("No new entry signals.")

    return "\n".join(lines)


# =========================================================
# Main Engine
# =========================================================
def main():
    ensure_dirs()

    positions = load_positions()
    decision_logs = []
    market_rows = []
    new_signals = []

    latest_market = {}

    # 1. Scan market first
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

            if df.empty or len(df) < 300:
                continue

            df = add_indicators(df)
            market = detect_market_state(df)
            last = df.iloc[-1]

            latest_market[sym] = {"df": df, "market": market}

            market_rows.append({
                "symbol": sym,
                "market": market,
                "close": round(float(last["close"]), 8),
                "rsi7": round(float(last["rsi7"]), 4),
                "vwap_dev_pct": round(float(last["vwap_dev"]) * 100, 4),
                "ema_dev_pct": round(float(last["ema_dev"]) * 100, 4),
            })

        except Exception as e:
            decision_logs.append({
                "time": utc_now_str(),
                "symbol": sym,
                "type": "SYSTEM",
                "action": "ERROR",
                "reason": str(e),
            })

    # 2. Update existing open positions
    for symbol, pos in list(positions.items()):
        if pos.get("status") != "OPEN":
            continue

        if symbol not in latest_market:
            decision_logs.append({
                "time": utc_now_str(),
                "symbol": symbol,
                "type": "POSITION",
                "action": "SKIP",
                "reason": "缺少最新市场数据",
            })
            continue

        updated_pos, decision = update_position(
            symbol,
            pos,
            latest_market[symbol]["df"],
            latest_market[symbol]["market"]
        )

        positions[symbol] = updated_pos

        decision_logs.append({
            "time": utc_now_str(),
            "symbol": symbol,
            "type": "POSITION",
            "action": decision["action"],
            "reason": decision["reason"],
            "current_price": decision["current_price"],
            "pnl_pct": decision["pnl_pct"],
            "market": decision["market"],
            "strategy": decision["strategy"],
        })

    # 3. Generate new entries only if open slots available
    open_count = sum(1 for p in positions.values() if p.get("status") == "OPEN")
    slots_left = max(CONFIG["max_open_positions"] - open_count, 0)

    if slots_left > 0:
        candidates = []

        for sym, pack in latest_market.items():
            if sym in positions and positions[sym].get("status") == "OPEN":
                continue

            signal = build_entry_signal(sym, pack["df"], pack["market"])
            if signal is not None:
                candidates.append(signal)

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        for signal in candidates[:slots_left]:
            side = signal["action"]
            positions[signal["symbol"]] = {
                "symbol": signal["symbol"],
                "side": side,
                "strategy": signal["strategy"],
                "market": signal["market"],
                "entry": signal["entry"],
                "tp": signal["tp"],
                "sl": signal["sl"],
                "size": 1.0,
                "status": "OPEN",
                "partial_taken": False,
                "opened_at": signal["opened_at"],
                "last_score": signal["score"],
            }

            new_signals.append(signal)

            decision_logs.append({
                "time": utc_now_str(),
                "symbol": signal["symbol"],
                "type": "ENTRY",
                "action": "OPEN",
                "reason": f"{signal['market']} + {signal['strategy']} 信号成立",
                "current_price": signal["entry"],
                "pnl_pct": 0.0,
                "market": signal["market"],
                "strategy": signal["strategy"],
            })

    # 4. Save state
    save_positions(positions)
    append_decision_logs(decision_logs)

    # 5. Output reports
    pd.DataFrame(new_signals).to_csv(SIGNALS_PATH, index=False, encoding="utf-8-sig")

    summary_text = build_summary(
        market_rows=market_rows,
        open_positions=positions,
        decision_rows=decision_logs,
        new_signals=new_signals
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("")
    print(f"saved: {POSITIONS_PATH}")
    print(f"saved: {SIGNALS_PATH}")
    print(f"saved: {DECISION_LOG_PATH}")
    print(f"saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
