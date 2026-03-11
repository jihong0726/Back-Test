import os
import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

VERSION = "V12_PAPER_TRADING_ENGINE"

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

    # 交易成本
    "fee_rate": 0.0006,
    "slippage_rate": 0.0002,

    # 简化资金费模型：每次引擎运行按名义仓位收取微小费用
    "funding_rate_per_cycle": 0.00002,

    # 风险参数
    "risk_mode": "standard",   # conservative / standard / aggressive
    "risk_per_trade": {
        "conservative": 0.005,  # 0.5%
        "standard": 0.01,       # 1%
        "aggressive": 0.02      # 2%
    },
    "max_margin_ratio": 0.25,   # 单笔最多使用净值 25% 作为保证金
    "default_leverage": 5,

    # 止盈止损
    "atr_stop_mult": 1.0,
    "atr_tp_mult": 1.8,
    "partial_tp_ratio": 0.5,
    "move_sl_after_profit_atr": 1.0,

    # 模拟账户
    "initial_balance": 10000.0,
}

STATE_DIR = "state"
REPORT_DIR = "reports"

POSITIONS_PATH = os.path.join(STATE_DIR, "positions.json")
ACCOUNT_PATH = os.path.join(STATE_DIR, "paper_account.json")

DECISION_LOG_PATH = os.path.join(REPORT_DIR, "decision_log.csv")
SIGNALS_PATH = os.path.join(REPORT_DIR, "signals.csv")
SUMMARY_PATH = os.path.join(REPORT_DIR, "latest_summary.txt")


# =========================================================
# 基础
# =========================================================
def ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def get_local_now():
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz)


def local_now_str() -> str:
    return get_local_now().strftime("%Y-%m-%d %H:%M:%S")


def get_next_run_info():
    now = get_local_now()

    minute = now.minute
    next_minute = ((minute // 5) + 1) * 5

    if next_minute >= 60:
        next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_run = now.replace(minute=next_minute, second=0, microsecond=0)

    delta = next_run - now
    minutes_left = int(delta.total_seconds() // 60)
    seconds_left = int(delta.total_seconds() % 60)

    return {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "next_run": next_run.strftime("%Y-%m-%d %H:%M:%S"),
        "minutes_left": minutes_left,
        "seconds_left": seconds_left
    }


def side_symbol(side: str) -> str:
    if side in ("LONG", "long"):
        return "🟢 多"
    if side in ("SHORT", "short"):
        return "🔴 空"
    return "⚪ 观望"


def safe_round(v, n=8):
    try:
        return round(float(v), n)
    except Exception:
        return None


# =========================================================
# 文件读写
# =========================================================
def load_json_file(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_positions() -> dict:
    data = load_json_file(POSITIONS_PATH, {})
    return data if isinstance(data, dict) else {}


def save_positions(positions: dict):
    save_json_file(POSITIONS_PATH, positions)


def default_account():
    init = float(CONFIG["initial_balance"])
    return {
        "initial_balance": init,
        "cash_balance": init,
        "equity": init,
        "used_margin": 0.0,
        "free_margin": init,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "funding_paid": 0.0,
        "total_fees": 0.0,
        "last_updated": local_now_str()
    }


def load_account() -> dict:
    data = load_json_file(ACCOUNT_PATH, default_account())
    if not isinstance(data, dict):
        return default_account()
    if not data:
        return default_account()
    return data


def save_account(account: dict):
    account["last_updated"] = local_now_str()
    save_json_file(ACCOUNT_PATH, account)


def append_decision_logs(rows: list[dict]):
    if not rows:
        return

    df = pd.DataFrame(rows)
    if os.path.exists(DECISION_LOG_PATH):
        old = pd.read_csv(DECISION_LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(DECISION_LOG_PATH, index=False, encoding="utf-8-sig")


# =========================================================
# Telegram
# =========================================================
def send_telegram_message(text: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets not found, skip sending message.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:3800]
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        print("Telegram message sent.")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def build_telegram_text(account: dict, new_signals: list[dict], decision_rows: list[dict], open_positions: dict):
    run_info = get_next_run_info()

    lines = []
    lines.append("📊 模拟交易引擎 V12")
    lines.append(f"当前时间：{run_info['now']}")
    lines.append(f"下一次建议时间：{run_info['next_run']}")
    lines.append(f"距离下一次建议：约 {run_info['minutes_left']} 分 {run_info['seconds_left']} 秒")
    lines.append("")

    lines.append("账户概况")
    lines.append(f"初始资金：{account['initial_balance']:.2f}U")
    lines.append(f"当前净值：{account['equity']:.2f}U")
    lines.append(f"可用余额：{account['free_margin']:.2f}U")
    lines.append(f"已用保证金：{account['used_margin']:.2f}U")
    lines.append(f"已实现盈亏：{account['realized_pnl']:.2f}U")
    lines.append(f"未实现盈亏：{account['unrealized_pnl']:.2f}U")
    lines.append(f"手续费：{account['total_fees']:.2f}U")
    lines.append(f"资金费：{account['funding_paid']:.2f}U")
    lines.append("")

    lines.append("━━━━━━━━━━━━━━")
    lines.append("📌 当前仓位信息")
    lines.append("━━━━━━━━━━━━━━")
    lines.append("")

    open_rows = []
    for symbol, pos in open_positions.items():
        if pos.get("status") == "OPEN":
            open_rows.append(pos)

    if open_rows:
        for pos in open_rows[:10]:
            lines.append(f"{pos['symbol']}  {side_symbol(pos.get('side'))}")
            lines.append(f"策略：{pos.get('strategy', '-')}")
            lines.append(f"入场：{pos.get('entry')}")
            lines.append(f"止盈：{pos.get('tp')}")
            lines.append(f"止损：{pos.get('sl')}")
            lines.append(f"杠杆：{pos.get('leverage', '-')}x")
            lines.append(f"保证金：{float(pos.get('margin', 0) or 0):.2f}U")
            lines.append(f"名义仓位：{float(pos.get('notional', 0) or 0):.2f}U")
            lines.append("")
    else:
        lines.append("目前没有持仓")
        lines.append("")

    entry_rows = [x for x in decision_rows if x.get("type") == "ENTRY" and x.get("action") == "OPEN"]
    close_rows = [x for x in decision_rows if x.get("type") == "POSITION" and x.get("action") == "CLOSE"]
    manage_rows = [x for x in decision_rows if x.get("type") == "POSITION" and x.get("action") in ("MOVE_SL", "PARTIAL_TP")]

    if entry_rows:
        lines.append("━━━━━━━━━━━━━━")
        lines.append("🟢 本轮新开仓")
        lines.append("━━━━━━━━━━━━━━")
        lines.append("")
        for r in entry_rows[:5]:
            lines.append(f"{r['symbol']}")
            lines.append(f"方向：{side_symbol(r.get('side', ''))}")
            lines.append(f"策略：{r.get('strategy', '-')}")
            lines.append(f"原因：{r.get('reason', '')}")
            if r.get("entry") is not None:
                lines.append(f"入场：{r['entry']}")
            if r.get("tp") is not None:
                lines.append(f"止盈：{r['tp']}")
            if r.get("sl") is not None:
                lines.append(f"止损：{r['sl']}")
            if r.get("leverage") is not None:
                lines.append(f"杠杆：{r['leverage']}x")
            if r.get("margin") is not None:
                lines.append(f"保证金：{float(r['margin']):.2f}U")
            if r.get("notional") is not None:
                lines.append(f"名义仓位：{float(r['notional']):.2f}U")
            lines.append("")

    if manage_rows:
        lines.append("━━━━━━━━━━━━━━")
        lines.append("⚙️ 本轮持仓调整")
        lines.append("━━━━━━━━━━━━━━")
        lines.append("")
        for r in manage_rows[:5]:
            pnl = r.get("pnl_pct", 0.0)
            action_map = {
                "MOVE_SL": "移动止损",
                "PARTIAL_TP": "部分止盈"
            }
            lines.append(f"{r['symbol']}")
            lines.append(f"操作：{action_map.get(r['action'], r['action'])}")
            lines.append(f"当前收益：{pnl:.2f}%")
            lines.append(f"原因：{r.get('reason', '')}")
            lines.append("")

    if close_rows:
        lines.append("━━━━━━━━━━━━━━")
        lines.append("❌ 本轮已平仓")
        lines.append("━━━━━━━━━━━━━━")
        lines.append("")
        for r in close_rows[:5]:
            pnl_pct = r.get("pnl_pct", 0.0)
            pnl_usd = r.get("pnl_usd", 0.0)
            lines.append(f"{r['symbol']}")
            lines.append(f"结果：{r.get('reason', '')}")
            lines.append(f"收益率：{pnl_pct:.2f}%")
            lines.append(f"盈亏：{pnl_usd:.2f}U")
            lines.append("")

    if not entry_rows and not manage_rows and not close_rows:
        lines.append("本轮没有新的交易变化")

    return "\n".join(lines)


# =========================================================
# 数据抓取
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
# 指标
# =========================================================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        )
    )
    return tr.rolling(period, min_periods=period).mean().bfill()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]

    cum_pv = pv.cumsum()
    cum_vol = df["volume"].replace(0, np.nan).cumsum()

    vwap = cum_pv / cum_vol
    return vwap.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def add_strategy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["rsi7"] = calc_rsi(df["close"], 7)
    df["rsi14"] = calc_rsi(df["close"], 14)
    df["atr14"] = calc_atr(df, 14)
    df["vwap"] = calc_vwap(df)

    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["ema_dev"] = (df["close"] - df["ema20"]) / df["ema20"]
    df["ema_spread"] = (df["ema20"] - df["ema50"]) / df["close"]
    df["atr_ratio"] = df["atr14"] / df["close"]

    return df


# =========================================================
# 市场状态
# =========================================================
def detect_market_regime(df: pd.DataFrame) -> str:
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
# 策略
# =========================================================
def calc_signal_score(market: str, action: str, vwap_dev: float, ema_dev: float, rsi7: float) -> float:
    score = 50.0
    score += min(abs(vwap_dev) * 1500, 30)
    score += min(abs(ema_dev) * 600, 15)

    if market == "RANGE":
        score += 8
    elif market in ("TREND_UP", "TREND_DOWN"):
        score += 5

    if action == "long":
        if rsi7 < 20:
            score += 10
        elif rsi7 < 30:
            score += 6
        elif rsi7 < 40:
            score += 3

    if action == "short":
        if rsi7 > 80:
            score += 10
        elif rsi7 > 70:
            score += 6
        elif rsi7 > 60:
            score += 3

    return round(min(score, 100), 2)


def strategy_decision(df: pd.DataFrame) -> dict:
    df = add_strategy_indicators(df)

    if len(df) < 100:
        return {
            "action": "neutral",
            "market": "UNKNOWN",
            "strategy": None,
            "score": 0,
            "entry": None,
            "tp": None,
            "sl": None,
            "reason": "数据不足"
        }

    last = df.iloc[-1]

    price = float(last["close"])
    vwap = float(last["vwap"])
    ema20 = float(last["ema20"])
    atr = float(last["atr14"])
    rsi7 = float(last["rsi7"])
    vwap_dev = float(last["vwap_dev"])
    ema_dev = float(last["ema_dev"])

    market = detect_market_regime(df)

    action = "neutral"
    strategy = None
    tp = None
    sl = None
    reason = "暂无有效信号"
    score = 0.0

    if market == "RANGE":
        if vwap_dev <= -0.025 and rsi7 < 35:
            action = "long"
            strategy = "VWAP_2.5"
            tp = vwap
            sl = price - atr
            reason = "震荡市场中价格显著低于 VWAP，且 RSI 偏低，做多等回归"

        elif vwap_dev >= 0.025 and rsi7 > 65:
            action = "short"
            strategy = "VWAP_2.5"
            tp = vwap
            sl = price + atr
            reason = "震荡市场中价格显著高于 VWAP，且 RSI 偏高，做空等回归"

        elif vwap_dev <= -0.020 and rsi7 < 40:
            action = "long"
            strategy = "VWAP_2.0"
            tp = vwap
            sl = price - atr
            reason = "震荡市场中价格低于 VWAP 2%，偏离足够，做多等回归"

        elif vwap_dev >= 0.020 and rsi7 > 60:
            action = "short"
            strategy = "VWAP_2.0"
            tp = vwap
            sl = price + atr
            reason = "震荡市场中价格高于 VWAP 2%，偏离足够，做空等回归"

        elif vwap_dev <= -0.015 and rsi7 < 35:
            action = "long"
            strategy = "VWAP_1.5"
            tp = vwap
            sl = price - atr
            reason = "震荡市场中价格低于 VWAP 1.5%，轻仓做多等回归"

        elif vwap_dev >= 0.015 and rsi7 > 65:
            action = "short"
            strategy = "VWAP_1.5"
            tp = vwap
            sl = price + atr
            reason = "震荡市场中价格高于 VWAP 1.5%，轻仓做空等回归"

    elif market == "TREND_UP":
        if price < ema20 and rsi7 < 40:
            action = "long"
            strategy = "TrendPullback_Long"
            tp = price + atr * CONFIG["atr_tp_mult"]
            sl = price - atr * CONFIG["atr_stop_mult"]
            reason = "上升趋势中回踩 EMA20，RSI 偏低，顺势做多"

    elif market == "TREND_DOWN":
        if price > ema20 and rsi7 > 60:
            action = "short"
            strategy = "TrendPullback_Short"
            tp = price - atr * CONFIG["atr_tp_mult"]
            sl = price + atr * CONFIG["atr_stop_mult"]
            reason = "下降趋势中反弹到 EMA20 上方，RSI 偏高，顺势做空"

    if action != "neutral":
        score = calc_signal_score(market, action, vwap_dev, ema_dev, rsi7)

    return {
        "action": action,
        "market": market,
        "strategy": strategy,
        "score": score,
        "entry": safe_round(price, 8) if action != "neutral" else None,
        "tp": safe_round(tp, 8) if tp is not None else None,
        "sl": safe_round(sl, 8) if sl is not None else None,
        "reason": reason,
        "rsi7": safe_round(rsi7, 4),
        "vwap_dev_pct": safe_round(vwap_dev * 100, 4),
        "ema_dev_pct": safe_round(ema_dev * 100, 4),
        "atr": safe_round(atr, 8),
    }


# =========================================================
# 模拟账户
# =========================================================
def compute_position_size(account: dict, entry: float, sl: float):
    equity = float(account["equity"])
    risk_ratio = float(CONFIG["risk_per_trade"][CONFIG["risk_mode"]])

    risk_amount = equity * risk_ratio
    stop_distance_pct = abs(entry - sl) / max(entry, 1e-9)

    if stop_distance_pct <= 0:
        return None

    notional = risk_amount / stop_distance_pct
    leverage = float(CONFIG["default_leverage"])
    margin = notional / leverage

    max_margin = equity * float(CONFIG["max_margin_ratio"])
    if margin > max_margin:
        margin = max_margin
        notional = margin * leverage

    if margin > float(account["free_margin"]):
        margin = float(account["free_margin"]) * 0.95
        notional = margin * leverage

    if margin <= 0 or notional <= 0:
        return None

    return {
        "risk_amount": safe_round(risk_amount, 4),
        "stop_distance_pct": safe_round(stop_distance_pct * 100, 4),
        "margin": safe_round(margin, 4),
        "notional": safe_round(notional, 4),
        "leverage": leverage
    }


def calculate_unrealized_pnl(side: str, entry: float, current: float, notional: float | None):
    notional = float(notional or 0)
    if notional <= 0:
        return 0.0

    qty = notional / max(entry, 1e-9)
    if side == "LONG":
        return (current - entry) * qty
    return (entry - current) * qty


def update_account_snapshot(account: dict, positions: dict, latest_market: dict):
    unrealized = 0.0
    used_margin = 0.0

    for symbol, pos in positions.items():
        if pos.get("status") != "OPEN":
            continue
        if symbol not in latest_market:
            continue

        current = float(latest_market[symbol]["df"].iloc[-1]["close"])
        entry = float(pos.get("entry", 0))
        notional = float(pos.get("notional", 0) or 0)
        margin = float(pos.get("margin", 0) or 0)
        side = pos.get("side", "")

        unrealized += calculate_unrealized_pnl(side, entry, current, notional)
        used_margin += margin

    account["used_margin"] = safe_round(used_margin, 4)
    account["unrealized_pnl"] = safe_round(unrealized, 4)
    account["equity"] = safe_round(account["cash_balance"] + unrealized, 4)
    account["free_margin"] = safe_round(account["equity"] - used_margin, 4)


def apply_open_fee(account: dict, notional: float):
    fee = notional * float(CONFIG["fee_rate"] + CONFIG["slippage_rate"])
    account["cash_balance"] -= fee
    account["total_fees"] += fee
    return fee


def apply_funding(account: dict, positions: dict):
    funding_rate = float(CONFIG["funding_rate_per_cycle"])
    total = 0.0

    for _, pos in positions.items():
        if pos.get("status") != "OPEN":
            continue
        notional = float(pos.get("notional", 0) or 0)
        fee = notional * funding_rate
        account["cash_balance"] -= fee
        account["funding_paid"] += fee
        total += fee

    return total


# =========================================================
# 持仓管理
# =========================================================
def current_pnl_pct(side: str, entry: float, current: float) -> float:
    if side == "LONG":
        return (current / entry - 1) * 100
    return (entry / current - 1) * 100


def close_position(account: dict, pos: dict, current: float, reason: str):
    side = pos.get("side", "")
    entry = float(pos.get("entry", 0) or 0)
    notional = float(pos.get("notional", 0) or 0)

    if entry <= 0 or notional <= 0:
        pnl_after_fee = 0.0
    else:
        qty = notional / max(entry, 1e-9)

        if side == "LONG":
            pnl = (current - entry) * qty
        else:
            pnl = (entry - current) * qty

        close_fee = notional * float(CONFIG["fee_rate"] + CONFIG["slippage_rate"])
        pnl_after_fee = pnl - close_fee
        account["total_fees"] += close_fee

    account["cash_balance"] += pnl_after_fee
    account["realized_pnl"] += pnl_after_fee

    pos["status"] = "CLOSED"
    pos["closed_at"] = local_now_str()
    pos["exit_price"] = safe_round(current, 8)
    pos["exit_reason"] = reason
    pos["realized_pnl"] = safe_round(pnl_after_fee, 4)

    return pnl_after_fee


def update_position(symbol: str, pos: dict, df: pd.DataFrame, market: str, account: dict):
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
        "current_price": safe_round(current, 8),
        "pnl_pct": safe_round(current_pnl_pct(side, entry, current), 4),
        "pnl_usd": safe_round(
            calculate_unrealized_pnl(side, entry, current, float(pos.get("notional", 0) or 0)),
            4
        ),
        "market": market,
        "strategy": pos.get("strategy", "-"),
        "side": side,
        "entry": entry,
        "tp": tp,
        "sl": sl,
    }

    if status != "OPEN":
        result["action"] = "CLOSED"
        result["reason"] = "position already closed"
        return pos, result

    if side == "LONG":
        if current <= sl:
            pnl_after_fee = close_position(account, pos, current, "STOP_LOSS")
            result["action"] = "CLOSE"
            result["reason"] = "触发止损"
            result["pnl_usd"] = safe_round(pnl_after_fee, 4)
            return pos, result

        if current >= tp:
            pnl_after_fee = close_position(account, pos, current, "TAKE_PROFIT")
            result["action"] = "CLOSE"
            result["reason"] = "触发止盈"
            result["pnl_usd"] = safe_round(pnl_after_fee, 4)
            return pos, result

    else:
        if current >= sl:
            pnl_after_fee = close_position(account, pos, current, "STOP_LOSS")
            result["action"] = "CLOSE"
            result["reason"] = "触发止损"
            result["pnl_usd"] = safe_round(pnl_after_fee, 4)
            return pos, result

        if current <= tp:
            pnl_after_fee = close_position(account, pos, current, "TAKE_PROFIT")
            result["action"] = "CLOSE"
            result["reason"] = "触发止盈"
            result["pnl_usd"] = safe_round(pnl_after_fee, 4)
            return pos, result

    if not partial_taken:
        if side == "LONG" and current >= entry + atr:
            pos["partial_taken"] = True
            pos["size"] = safe_round(size * (1 - CONFIG["partial_tp_ratio"]), 4)
            pos["sl"] = safe_round(max(sl, entry), 8)
            result["action"] = "PARTIAL_TP"
            result["reason"] = "达到第一目标，部分止盈并上移止损到入场附近"
            return pos, result

        if side == "SHORT" and current <= entry - atr:
            pos["partial_taken"] = True
            pos["size"] = safe_round(size * (1 - CONFIG["partial_tp_ratio"]), 4)
            pos["sl"] = safe_round(min(sl, entry), 8)
            result["action"] = "PARTIAL_TP"
            result["reason"] = "达到第一目标，部分止盈并下移止损到入场附近"
            return pos, result

    move_trigger = float(CONFIG["move_sl_after_profit_atr"]) * atr
    if side == "LONG" and current >= entry + move_trigger:
        new_sl = safe_round(max(sl, ema20, entry), 8)
        if new_sl > sl:
            pos["sl"] = new_sl
            result["action"] = "MOVE_SL"
            result["reason"] = "盈利扩大，上移止损"
            return pos, result

    if side == "SHORT" and current <= entry - move_trigger:
        new_sl = safe_round(min(sl, ema20, entry), 8)
        if new_sl < sl:
            pos["sl"] = new_sl
            result["action"] = "MOVE_SL"
            result["reason"] = "盈利扩大，下移止损"
            return pos, result

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
# 摘要
# =========================================================
def build_summary(account: dict, market_rows: list[dict], open_positions: dict, decision_rows: list[dict], new_signals: list[dict]):
    lines = []
    lines.append(f"Version: {VERSION}")
    lines.append(f"Generated At: {local_now_str()}")
    lines.append("")

    lines.append("=== ACCOUNT ===")
    lines.append(pd.DataFrame([account]).to_string(index=False))
    lines.append("")

    lines.append("=== MARKET STATES ===")
    if market_rows:
        lines.append(pd.DataFrame(market_rows).to_string(index=False))
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
                "margin": pos.get("margin"),
                "notional": pos.get("notional"),
                "leverage": pos.get("leverage"),
                "size": pos.get("size"),
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
# 主程序
# =========================================================
def main():
    ensure_dirs()

    positions = load_positions()
    account = load_account()

    decision_logs = []
    market_rows = []
    new_signals = []
    latest_market = {}

    # 扫市场
    for sym in CONFIG["symbols"]:
        try:
            print("processing", sym)
            raw_df = fetch_bitget(
                sym,
                interval=CONFIG["interval"],
                limit=CONFIG["bars"],
                page_limit=CONFIG["page_limit"],
                max_pages=CONFIG["max_pages"]
            )

            if raw_df.empty or len(raw_df) < 300:
                continue

            decision = strategy_decision(raw_df)
            latest_market[sym] = {
                "df": add_strategy_indicators(raw_df),
                "market": decision["market"],
                "decision": decision,
            }

            market_rows.append({
                "symbol": sym,
                "market": decision["market"],
                "close": decision["entry"],
                "rsi7": decision.get("rsi7"),
                "vwap_dev_pct": decision.get("vwap_dev_pct"),
                "ema_dev_pct": decision.get("ema_dev_pct"),
                "score": decision.get("score", 0),
            })

        except Exception as e:
            decision_logs.append({
                "time": local_now_str(),
                "symbol": sym,
                "type": "SYSTEM",
                "action": "ERROR",
                "reason": str(e),
            })

    # 收资金费
    funding_paid = apply_funding(account, positions)
    if funding_paid > 0:
        decision_logs.append({
            "time": local_now_str(),
            "symbol": "ACCOUNT",
            "type": "ACCOUNT",
            "action": "FUNDING",
            "reason": f"本轮收取资金费 {funding_paid:.4f}U",
        })

    # 更新持仓
    for symbol, pos in list(positions.items()):
        if pos.get("status") != "OPEN":
            continue

        if symbol not in latest_market:
            decision_logs.append({
                "time": local_now_str(),
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
            latest_market[symbol]["market"],
            account
        )

        positions[symbol] = updated_pos

        decision_logs.append({
            "time": local_now_str(),
            "symbol": symbol,
            "type": "POSITION",
            "action": decision["action"],
            "reason": decision["reason"],
            "current_price": decision["current_price"],
            "pnl_pct": decision["pnl_pct"],
            "pnl_usd": decision["pnl_usd"],
            "market": decision["market"],
            "strategy": decision["strategy"],
            "side": decision["side"],
            "entry": decision["entry"],
            "tp": decision["tp"],
            "sl": decision["sl"],
        })

    # 账户快照
    update_account_snapshot(account, positions, latest_market)

    # 新开仓
    open_count = sum(1 for p in positions.values() if p.get("status") == "OPEN")
    slots_left = max(int(CONFIG["max_open_positions"]) - open_count, 0)

    if slots_left > 0:
        candidates = []

        for sym, pack in latest_market.items():
            if sym in positions and positions[sym].get("status") == "OPEN":
                continue

            signal = pack["decision"]
            if signal["action"] == "neutral":
                continue
            if signal["score"] < CONFIG["min_signal_score"]:
                continue

            candidates.append({
                "symbol": sym,
                **signal
            })

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        for signal in candidates[:slots_left]:
            sizing = compute_position_size(account, signal["entry"], signal["sl"])
            if sizing is None:
                continue

            open_fee = apply_open_fee(account, sizing["notional"])

            side = "LONG" if signal["action"] == "long" else "SHORT"

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
                "opened_at": local_now_str(),
                "last_score": signal["score"],
                "margin": sizing["margin"],
                "notional": sizing["notional"],
                "leverage": sizing["leverage"],
                "risk_amount": sizing["risk_amount"],
                "stop_distance_pct": sizing["stop_distance_pct"],
            }

            new_signals.append({
                "symbol": signal["symbol"],
                "market": signal["market"],
                "strategy": signal["strategy"],
                "action": side,
                "entry": signal["entry"],
                "tp": signal["tp"],
                "sl": signal["sl"],
                "score": signal["score"],
                "reason": signal["reason"],
                "margin": sizing["margin"],
                "notional": sizing["notional"],
                "leverage": sizing["leverage"],
                "risk_amount": sizing["risk_amount"],
                "open_fee": safe_round(open_fee, 4),
            })

            decision_logs.append({
                "time": local_now_str(),
                "symbol": signal["symbol"],
                "type": "ENTRY",
                "action": "OPEN",
                "reason": signal["reason"],
                "current_price": signal["entry"],
                "pnl_pct": 0.0,
                "pnl_usd": 0.0,
                "market": signal["market"],
                "strategy": signal["strategy"],
                "side": side,
                "entry": signal["entry"],
                "tp": signal["tp"],
                "sl": signal["sl"],
                "margin": sizing["margin"],
                "notional": sizing["notional"],
                "leverage": sizing["leverage"],
            })

    # 再更新账户
    update_account_snapshot(account, positions, latest_market)

    # 保存
    save_positions(positions)
    save_account(account)
    append_decision_logs(decision_logs)

    pd.DataFrame(new_signals).to_csv(SIGNALS_PATH, index=False, encoding="utf-8-sig")

    summary_text = build_summary(
        account=account,
        market_rows=market_rows,
        open_positions=positions,
        decision_rows=decision_logs,
        new_signals=new_signals
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    telegram_text = build_telegram_text(account, new_signals, decision_logs, positions)
    send_telegram_message(telegram_text)

    print(summary_text)
    print("")
    print(f"saved: {POSITIONS_PATH}")
    print(f"saved: {ACCOUNT_PATH}")
    print(f"saved: {SIGNALS_PATH}")
    print(f"saved: {DECISION_LOG_PATH}")
    print(f"saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
