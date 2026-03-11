import os
import json
import time
import math
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import unicodedata

VERSION = "V8.3_市场状态_滚动回测_风控版"
REPORT_DIR = "reports"

CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"],
    "interval": "5m",
    "target_bars": 3000,
    "request_limit": 1000,
    "max_pages": 8,
    "fee_rate": 0.0006,
    "slippage_rate": 0.0002,
    "min_bars_required": 1500,
    "walk_forward_train": 1000,
    "walk_forward_test": 300,
    "walk_forward_step": 300,
    "atr_stop_mult": 1.5,
    "atr_take_mult": 2.0,
}


# =========================================================
# 基础工具
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def text_width(s: str) -> int:
    s = str(s)
    width = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("F", "W", "A"):
            width += 2
        else:
            width += 1
    return width


def pad_text(s: str, width: int, align: str = "left") -> str:
    s = str(s)
    real = text_width(s)
    pad = max(width - real, 0)
    if align == "right":
        return " " * pad + s
    if align == "center":
        left = pad // 2
        right = pad - left
        return " " * left + s + " " * right
    return s + " " * pad


def format_table(df: pd.DataFrame, right_align_cols=None) -> str:
    if df is None or df.empty:
        return "暂无数据"

    if right_align_cols is None:
        right_align_cols = set()

    cols = list(df.columns)
    widths = []
    for c in cols:
        col_values = [str(c)] + [str(v) for v in df[c].tolist()]
        widths.append(max(text_width(v) for v in col_values) + 2)

    header = " | ".join(
        pad_text(c, widths[i], "center") for i, c in enumerate(cols)
    )
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))

    lines = [header, sep]
    for _, row in df.iterrows():
        row_cells = []
        for i, c in enumerate(cols):
            align = "right" if c in right_align_cols else "left"
            row_cells.append(pad_text(row[c], widths[i], align))
        lines.append(" | ".join(row_cells))

    return "\n".join(lines)


def safe_num(x, n=2):
    return f"{x:.{n}f}" if pd.notna(x) else "N/A"


def annual_factor_from_interval(interval: str) -> float:
    interval = interval.lower()
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        bars_per_day = (24 * 60) / minutes
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        bars_per_day = 24 / hours
    else:
        bars_per_day = 288
    return 252 * bars_per_day


def interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1H": 60 * 60_000,
        "4H": 4 * 60 * 60_000,
        "6H": 6 * 60 * 60_000,
        "12H": 12 * 60 * 60_000,
        "1D": 24 * 60 * 60_000,
    }
    return mapping.get(interval, 5 * 60_000)


# =========================================================
# 数据抓取
# =========================================================
def fetch_candles_bitget(symbol="BTCUSDT", interval="5m", target_bars=3000, request_limit=1000, max_pages=8):
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    step_ms = interval_to_ms(interval)

    end_time = int(time.time() * 1000)
    all_rows = []
    seen_ts = set()

    print(f"[{symbol}] 开始抓取，目标K线数: {target_bars}")

    for page in range(max_pages):
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "endTime": str(end_time),
            "limit": str(min(request_limit, 1000))
        }

        try:
            res = requests.get(url, params=params, timeout=20)
            res.raise_for_status()
            payload = res.json()
            data = payload.get("data", [])

            if not data:
                print(f"[{symbol}] 第 {page + 1} 页无数据，停止。")
                break

            batch = []
            for row in data:
                ts = int(row[0])
                if ts not in seen_ts:
                    seen_ts.add(ts)
                    batch.append(row)

            if not batch:
                print(f"[{symbol}] 第 {page + 1} 页全部重复，停止。")
                break

            all_rows.extend(batch)
            oldest_ts = min(int(r[0]) for r in batch)
            newest_ts = max(int(r[0]) for r in batch)

            print(f"[{symbol}] 第 {page + 1} 页新增 {len(batch)} 根，时间范围: {oldest_ts} ~ {newest_ts}，累计 {len(all_rows)} 根")

            if len(all_rows) >= target_bars:
                break

            end_time = oldest_ts - step_ms
            time.sleep(0.12)

        except Exception as e:
            print(f"[{symbol}] 抓取失败: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "base_vol", "quote_vol"]
    )

    for col in ["timestamp", "open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# =========================================================
# 技术指标
# =========================================================
def calc_rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calc_atr(df: pd.DataFrame, period=14):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return tr, atr


def calc_bollinger(series: pd.Series, period=20, std_mult=2.0):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + sd * std_mult
    lower = ma - sd * std_mult
    return ma, upper, lower


def calc_vwap(df: pd.DataFrame):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["base_vol"]
    cum_pv = pv.cumsum()
    cum_vol = df["base_vol"].replace(0, np.nan).cumsum()
    return (cum_pv / cum_vol).fillna(method="ffill")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()

    for span in [5, 9, 12, 20, 21, 34, 50, 55, 100, 200]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    for rsi_period in [7, 14]:
        df[f"rsi_{rsi_period}"] = calc_rsi(df["close"], rsi_period)

    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["close"])
    df["tr"], df["atr_14"] = calc_atr(df, 14)
    df["atr_5"] = df["tr"].rolling(5).mean()

    df["vol_ma_20"] = df["base_vol"].rolling(20).mean()
    df["vol_ma_50"] = df["base_vol"].rolling(50).mean()

    bb_mid, bb_upper, bb_lower = calc_bollinger(df["close"], 20, 2.0)
    df["bb_mid_20"] = bb_mid
    df["bb_upper_20"] = bb_upper
    df["bb_lower_20"] = bb_lower

    df["donchian_high_20"] = df["high"].shift(1).rolling(20).max()
    df["donchian_low_20"] = df["low"].shift(1).rolling(20).min()
    df["donchian_high_55"] = df["high"].shift(1).rolling(55).max()
    df["donchian_low_55"] = df["low"].shift(1).rolling(55).min()

    df["vwap"] = calc_vwap(df)

    df["ret"] = df["close"].pct_change().fillna(0)
    df["dist_ema20"] = (df["close"] / df["ema_20"]) - 1
    df["dist_ema50"] = (df["close"] / df["ema_50"]) - 1
    df["dist_vwap"] = (df["close"] / df["vwap"]) - 1

    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    temp = df.copy().set_index("timestamp")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "base_vol": "sum",
        "quote_vol": "sum"
    }

    df4h = temp.resample("4h").agg(agg).dropna().reset_index()
    df4h = add_indicators(df4h)
    return df4h


# =========================================================
# 市场状态识别
# =========================================================
def classify_regime(df: pd.DataFrame) -> pd.Series:
    ema_gap = ((df["ema_20"] - df["ema_50"]).abs() / df["close"]).fillna(0)
    ema20_slope = (df["ema_20"] - df["ema_20"].shift(5)) / df["close"]
    atr_ratio = (df["atr_14"] / df["close"]).fillna(0)

    regime = pd.Series("RANGE", index=df.index)

    trend_up = (df["ema_20"] > df["ema_50"]) & (ema20_slope > 0.002) & (ema_gap > 0.003)
    trend_down = (df["ema_20"] < df["ema_50"]) & (ema20_slope < -0.002) & (ema_gap > 0.003)
    high_vol_range = (atr_ratio > 0.01) & ~(trend_up | trend_down)

    regime[trend_up] = "TREND_UP"
    regime[trend_down] = "TREND_DOWN"
    regime[high_vol_range] = "RANGE_VOL"

    return regime


# =========================================================
# 策略生成
# =========================================================
def signal_from_conditions(long_cond, short_cond):
    return pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)))


def apply_regime_filter(signal: pd.Series, regime: pd.Series, allowed_regimes):
    filtered = signal.copy()
    mask = ~regime.isin(allowed_regimes)
    filtered[mask] = 0
    return filtered


def generate_strategies(df: pd.DataFrame):
    regime = classify_regime(df)
    strategies = {}

    # 反转类
    strategies["RSI反转_20_80"] = apply_regime_filter(
        signal_from_conditions(df["rsi_7"] < 20, df["rsi_7"] > 80),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["RSI反转_25_75"] = apply_regime_filter(
        signal_from_conditions(df["rsi_7"] < 25, df["rsi_7"] > 75),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["偏离EMA20_反转_0.015"] = apply_regime_filter(
        signal_from_conditions(df["dist_ema20"] < -0.015, df["dist_ema20"] > 0.015),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["偏离EMA20_反转_0.02"] = apply_regime_filter(
        signal_from_conditions(df["dist_ema20"] < -0.02, df["dist_ema20"] > 0.02),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["Bollinger反转_20_2.0"] = apply_regime_filter(
        signal_from_conditions(df["close"] < df["bb_lower_20"], df["close"] > df["bb_upper_20"]),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["VWAP回归_1.5%"] = apply_regime_filter(
        signal_from_conditions(df["dist_vwap"] < -0.015, df["dist_vwap"] > 0.015),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["VWAP回归_2.0%"] = apply_regime_filter(
        signal_from_conditions(df["dist_vwap"] < -0.02, df["dist_vwap"] > 0.02),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    strategies["低位修复_RSI_MACD"] = apply_regime_filter(
        signal_from_conditions(
            (df["rsi_7"] < 30) & (df["macd_hist"] > df["macd_hist"].shift(1)),
            (df["rsi_7"] > 70) & (df["macd_hist"] < df["macd_hist"].shift(1))
        ),
        regime,
        ["RANGE", "RANGE_VOL"]
    )

    # 趋势类
    strategies["EMA交叉_20_50"] = apply_regime_filter(
        signal_from_conditions(df["ema_20"] > df["ema_50"], df["ema_20"] < df["ema_50"]),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["EMA交叉_50_200"] = apply_regime_filter(
        signal_from_conditions(df["ema_50"] > df["ema_200"], df["ema_50"] < df["ema_200"]),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["趋势过滤_EMA50_200_RSI55_45"] = apply_regime_filter(
        signal_from_conditions(
            (df["ema_50"] > df["ema_200"]) & (df["rsi_14"] > 55),
            (df["ema_50"] < df["ema_200"]) & (df["rsi_14"] < 45)
        ),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["Donchian突破_20"] = apply_regime_filter(
        signal_from_conditions(df["close"] > df["donchian_high_20"], df["close"] < df["donchian_low_20"]),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["Donchian突破_55"] = apply_regime_filter(
        signal_from_conditions(df["close"] > df["donchian_high_55"], df["close"] < df["donchian_low_55"]),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["ATR突破_EMA20_1.0"] = apply_regime_filter(
        signal_from_conditions(
            df["close"] > (df["ema_20"] + df["atr_14"] * 1.0),
            df["close"] < (df["ema_20"] - df["atr_14"] * 1.0)
        ),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    strategies["多因子共振_趋势版"] = apply_regime_filter(
        signal_from_conditions(
            (df["close"] > df["ema_200"]) &
            (df["ema_20"] > df["ema_50"]) &
            (df["macd_hist"] > 0) &
            (df["rsi_14"] > 55),
            (df["close"] < df["ema_200"]) &
            (df["ema_20"] < df["ema_50"]) &
            (df["macd_hist"] < 0) &
            (df["rsi_14"] < 45)
        ),
        regime,
        ["TREND_UP", "TREND_DOWN"]
    )

    cleaned = {}
    for k, v in strategies.items():
        sig = pd.Series(v, index=df.index).fillna(0)
        if sig.abs().sum() > 0:
            cleaned[k] = sig

    return cleaned, regime


# =========================================================
# 回测核心：带止盈止损
# =========================================================
def calc_equity_metrics(net_ret: pd.Series, annual_factor: float):
    net_ret = net_ret.fillna(0)
    equity = (1 + net_ret).cumprod()

    total_return = (equity.iloc[-1] - 1) * 100
    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd = drawdown.min() * 100

    avg_ret = net_ret.mean()
    std_ret = net_ret.std()
    sharpe = (avg_ret / std_ret * math.sqrt(annual_factor)) if std_ret and not np.isnan(std_ret) and std_ret != 0 else 0

    active_returns = net_ret[net_ret != 0]
    win_rate = ((active_returns > 0).sum() / len(active_returns) * 100) if len(active_returns) > 0 else 0

    return {
        "equity": equity,
        "total_return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate
    }


def run_backtest_with_stops(
    df: pd.DataFrame,
    signal: pd.Series,
    fee_rate=0.0006,
    slippage_rate=0.0002,
    annual_factor=72576,
    atr_stop_mult=1.5,
    atr_take_mult=2.0
):
    signal = pd.Series(signal, index=df.index).fillna(0)
    n = len(df)

    position = 0
    entry_price = None
    stop_price = None
    take_price = None

    returns = np.zeros(n)
    trade_count = 0
    signal_change_count = 0
    position_bars = 0

    for i in range(1, n):
        prev_close = float(df["close"].iloc[i - 1])
        current_open = float(df["open"].iloc[i])
        current_high = float(df["high"].iloc[i])
        current_low = float(df["low"].iloc[i])
        current_close = float(df["close"].iloc[i])
        atr = float(df["atr_14"].iloc[i]) if pd.notna(df["atr_14"].iloc[i]) else 0.0
        desired_signal = int(signal.iloc[i - 1])

        # 持仓期间
        if position != 0:
            position_bars += 1

            exit_price = None
            stop_hit = False
            take_hit = False

            if position == 1:
                if current_low <= stop_price:
                    exit_price = stop_price
                    stop_hit = True
                elif current_high >= take_price:
                    exit_price = take_price
                    take_hit = True
            elif position == -1:
                if current_high >= stop_price:
                    exit_price = stop_price
                    stop_hit = True
                elif current_low <= take_price:
                    exit_price = take_price
                    take_hit = True

            # 若命中止盈止损
            if exit_price is not None:
                gross_ret = position * ((exit_price / prev_close) - 1)
                cost = fee_rate + slippage_rate
                returns[i] = gross_ret - cost
                position = 0
                entry_price = None
                stop_price = None
                take_price = None
                trade_count += 1
                signal_change_count += 1
                continue

            # 正常按收盘计算
            gross_ret = position * ((current_close / prev_close) - 1)
            returns[i] = gross_ret

            # 反向信号则平仓并反手
            if desired_signal != 0 and desired_signal != position:
                returns[i] -= (fee_rate + slippage_rate)
                position = desired_signal
                entry_price = current_close
                stop_price = entry_price - atr_stop_mult * atr if position == 1 else entry_price + atr_stop_mult * atr
                take_price = entry_price + atr_take_mult * atr if position == 1 else entry_price - atr_take_mult * atr
                trade_count += 1
                signal_change_count += 1
                continue

        # 空仓时进场
        if position == 0 and desired_signal != 0 and atr > 0:
            position = desired_signal
            entry_price = current_open
            stop_price = entry_price - atr_stop_mult * atr if position == 1 else entry_price + atr_stop_mult * atr
            take_price = entry_price + atr_take_mult * atr if position == 1 else entry_price - atr_take_mult * atr
            returns[i] -= (fee_rate + slippage_rate)
            trade_count += 1
            signal_change_count += 1

    net_ret = pd.Series(returns, index=df.index)
    m = calc_equity_metrics(net_ret, annual_factor)
    exposure = position_bars / n * 100 if n > 0 else 0

    return {
        "净收益(%)": m["total_return"],
        "最大回撤(%)": m["max_dd"],
        "Sharpe": m["sharpe"],
        "胜率(%)": m["win_rate"],
        "交易次数": int(trade_count),
        "信号变更次数": int(signal_change_count),
        "持仓占比(%)": exposure,
        "最终资金曲线": m["equity"]
    }


def run_walk_forward_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    interval: str,
    fee_rate: float,
    slippage_rate: float,
    train_size: int,
    test_size: int,
    step_size: int,
    atr_stop_mult: float,
    atr_take_mult: float
):
    annual_factor = annual_factor_from_interval(interval)

    windows = []
    start = 0
    n = len(df)

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        df_train = df.iloc[start:train_end].copy()
        df_test = df.iloc[train_end:test_end].copy()

        sig_train = signal.iloc[start:train_end]
        sig_test = signal.iloc[train_end:test_end]

        train_res = run_backtest_with_stops(
            df_train, sig_train, fee_rate, slippage_rate, annual_factor,
            atr_stop_mult, atr_take_mult
        )
        test_res = run_backtest_with_stops(
            df_test, sig_test, fee_rate, slippage_rate, annual_factor,
            atr_stop_mult, atr_take_mult
        )

        windows.append({
            "train": train_res,
            "test": test_res,
            "start": start,
            "train_end": train_end,
            "test_end": test_end
        })

        start += step_size

    if not windows:
        return None

    def avg_metric(key, side):
        vals = [w[side][key] for w in windows]
        return float(np.mean(vals))

    full_res = run_backtest_with_stops(
        df, signal, fee_rate, slippage_rate, annual_factor,
        atr_stop_mult, atr_take_mult
    )

    return {
        "windows": windows,
        "avg_train_return_pct": avg_metric("净收益(%)", "train"),
        "avg_train_dd_pct": avg_metric("最大回撤(%)", "train"),
        "avg_train_sharpe": avg_metric("Sharpe", "train"),
        "avg_test_return_pct": avg_metric("净收益(%)", "test"),
        "avg_test_dd_pct": avg_metric("最大回撤(%)", "test"),
        "avg_test_sharpe": avg_metric("Sharpe", "test"),
        "avg_test_win_rate_pct": avg_metric("胜率(%)", "test"),
        "avg_test_trades": avg_metric("交易次数", "test"),
        "window_count": len(windows),
        "full": full_res
    }


# =========================================================
# 评分
# =========================================================
def compute_strategy_score(test_return, test_dd, test_sharpe, win_rate, trades, window_count):
    score = 0.0
    score += test_return * 1.6
    score += test_sharpe * 7.0
    score -= abs(test_dd) * 1.5
    score += (win_rate - 50) * 0.35
    score += window_count * 1.0

    if trades < 2:
        score -= 20
    elif trades < 5:
        score -= 8
    elif trades > 80:
        score -= 5

    return score


# =========================================================
# 单币对回测
# =========================================================
def backtest_one_symbol(symbol: str, cfg: dict):
    print(f"\n[{VERSION}] 开始处理 {symbol} ...")

    raw = fetch_candles_bitget(
        symbol=symbol,
        interval=cfg["interval"],
        target_bars=cfg["target_bars"],
        request_limit=cfg["request_limit"],
        max_pages=cfg["max_pages"]
    )

    if raw.empty or len(raw) < cfg["min_bars_required"]:
        print(f"[{symbol}] 数据不足，跳过。当前K线数: {len(raw)}")
        return None

    print(f"[{symbol}] 抓取完成，最终K线数: {len(raw)}")

    df = add_indicators(raw)
    df4h = resample_4h(df)
    strategies, regime = generate_strategies(df)

    rows = []

    for strategy_name, signal in strategies.items():
        res = run_walk_forward_backtest(
            df=df,
            signal=signal,
            interval=cfg["interval"],
            fee_rate=cfg["fee_rate"],
            slippage_rate=cfg["slippage_rate"],
            train_size=cfg["walk_forward_train"],
            test_size=cfg["walk_forward_test"],
            step_size=cfg["walk_forward_step"],
            atr_stop_mult=cfg["atr_stop_mult"],
            atr_take_mult=cfg["atr_take_mult"]
        )
        if res is None:
            continue

        score = compute_strategy_score(
            test_return=res["avg_test_return_pct"],
            test_dd=res["avg_test_dd_pct"],
            test_sharpe=res["avg_test_sharpe"],
            win_rate=res["avg_test_win_rate_pct"],
            trades=res["avg_test_trades"],
            window_count=res["window_count"]
        )

        full_r = res["full"]

        rows.append({
            "symbol": symbol,
            "strategy": strategy_name,
            "bars": len(df),
            "window_count": res["window_count"],
            "avg_train_return_pct": res["avg_train_return_pct"],
            "avg_train_dd_pct": res["avg_train_dd_pct"],
            "avg_train_sharpe": res["avg_train_sharpe"],
            "avg_test_return_pct": res["avg_test_return_pct"],
            "avg_test_dd_pct": res["avg_test_dd_pct"],
            "avg_test_sharpe": res["avg_test_sharpe"],
            "avg_test_win_rate_pct": res["avg_test_win_rate_pct"],
            "avg_test_trades": res["avg_test_trades"],
            "full_return_pct": full_r["净收益(%)"],
            "full_dd_pct": full_r["最大回撤(%)"],
            "full_sharpe": full_r["Sharpe"],
            "full_win_rate_pct": full_r["胜率(%)"],
            "full_trades": full_r["交易次数"],
            "score": score
        })

    if not rows:
        return None

    result_df = pd.DataFrame(rows).sort_values(by=["score", "avg_test_return_pct"], ascending=[False, False]).reset_index(drop=True)

    latest_regime = regime.iloc[-1] if len(regime) else "UNKNOWN"
    regime_counts = regime.value_counts().to_dict()

    snapshot = {
        "symbol": symbol,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": float(df["close"].iloc[-1]),
        "current_ema20": float(df["ema_20"].iloc[-1]),
        "current_ema50": float(df["ema_50"].iloc[-1]),
        "current_macd": float(df["macd"].iloc[-1]),
        "current_rsi7": float(df["rsi_7"].iloc[-1]),
        "current_vwap": float(df["vwap"].iloc[-1]),
        "bars": len(df),
        "interval": cfg["interval"],
        "latest_regime": latest_regime,
        "regime_distribution": regime_counts
    }

    if not df4h.empty:
        snapshot["4h_context"] = {
            "ema20": float(df4h["ema_20"].iloc[-1]),
            "ema50": float(df4h["ema_50"].iloc[-1]),
            "atr14": float(df4h["atr_14"].iloc[-1]),
            "rsi14": float(df4h["rsi_14"].iloc[-1])
        }

    return {
        "symbol": symbol,
        "result_df": result_df,
        "snapshot": snapshot
    }


# =========================================================
# 多币对聚合
# =========================================================
def aggregate_across_symbols(all_symbol_result_dfs):
    combined = pd.concat(all_symbol_result_dfs, ignore_index=True)

    grouped = combined.groupby("strategy").agg(
        symbols_tested=("symbol", "count"),
        avg_score=("score", "mean"),
        median_score=("score", "median"),
        avg_test_return_pct=("avg_test_return_pct", "mean"),
        median_test_return_pct=("avg_test_return_pct", "median"),
        avg_test_dd_pct=("avg_test_dd_pct", "mean"),
        avg_test_sharpe=("avg_test_sharpe", "mean"),
        avg_test_win_rate_pct=("avg_test_win_rate_pct", "mean"),
        avg_full_return_pct=("full_return_pct", "mean"),
        avg_full_dd_pct=("full_dd_pct", "mean"),
        avg_full_sharpe=("full_sharpe", "mean"),
        positive_test_symbols=("avg_test_return_pct", lambda x: int((x > 0).sum())),
        positive_sharpe_symbols=("avg_test_sharpe", lambda x: int((x > 0).sum())),
    ).reset_index()

    grouped["robustness_score"] = (
        grouped["avg_score"] * 0.85
        + grouped["positive_test_symbols"] * 4
        + grouped["positive_sharpe_symbols"] * 2
        - grouped["avg_test_dd_pct"].abs() * 1.2
    )

    grouped = grouped.sort_values(
        by=["robustness_score", "avg_test_return_pct", "avg_test_sharpe"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return combined, grouped


# =========================================================
# 输出
# =========================================================
def save_reports(combined_detail_df, overall_df, per_symbol_best_df, snapshots):
    ensure_dir(REPORT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_csv = os.path.join(REPORT_DIR, f"全部币对_全部策略明细_{ts}.csv")
    overall_csv = os.path.join(REPORT_DIR, f"多币对_综合策略排行榜_{ts}.csv")
    best_csv = os.path.join(REPORT_DIR, f"各币对_最佳策略_{ts}.csv")
    snapshot_json = os.path.join(REPORT_DIR, f"市场快照_{ts}.json")
    summary_txt = os.path.join(REPORT_DIR, f"综合回测总结_{ts}.txt")

    combined_detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")
    per_symbol_best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")

    with open(snapshot_json, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)

    return {
        "detail_csv": detail_csv,
        "overall_csv": overall_csv,
        "best_csv": best_csv,
        "snapshot_json": snapshot_json,
        "summary_txt": summary_txt,
    }


def build_summary_text(cfg, overall_df, per_symbol_best_df, snapshots, paths):
    lines = []
    lines.append(f"[{VERSION}]")
    lines.append(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"回测币对: {', '.join(cfg['symbols'])}")
    lines.append(f"周期: {cfg['interval']}")
    lines.append(f"目标K线数: {cfg['target_bars']}")
    lines.append(f"Walk Forward: train={cfg['walk_forward_train']} / test={cfg['walk_forward_test']} / step={cfg['walk_forward_step']}")
    lines.append(f"ATR 止损倍数: {cfg['atr_stop_mult']}")
    lines.append(f"ATR 止盈倍数: {cfg['atr_take_mult']}")
    lines.append("=" * 140)

    lines.append("【多币对综合最佳策略 Top 12】")
    show_overall = overall_df.head(12).copy().rename(columns={
        "strategy": "策略名称",
        "symbols_tested": "测试币数",
        "robustness_score": "综合鲁棒评分",
        "avg_test_return_pct": "平均滚动样本外收益(%)",
        "avg_test_dd_pct": "平均滚动样本外回撤(%)",
        "avg_test_sharpe": "平均滚动样本外Sharpe",
        "positive_test_symbols": "样本外盈利币数",
        "positive_sharpe_symbols": "正Sharpe币数",
    })

    show_overall = show_overall[[
        "策略名称", "测试币数", "综合鲁棒评分",
        "平均滚动样本外收益(%)", "平均滚动样本外回撤(%)",
        "平均滚动样本外Sharpe", "样本外盈利币数", "正Sharpe币数"
    ]].copy()

    for c in ["综合鲁棒评分", "平均滚动样本外收益(%)", "平均滚动样本外回撤(%)", "平均滚动样本外Sharpe"]:
        show_overall[c] = show_overall[c].map(lambda x: safe_num(x, 2))

    lines.append(format_table(
        show_overall,
        right_align_cols={"测试币数", "综合鲁棒评分", "平均滚动样本外收益(%)", "平均滚动样本外回撤(%)", "平均滚动样本外Sharpe", "样本外盈利币数", "正Sharpe币数"}
    ))

    lines.append("")
    lines.append("【各币对最佳策略】")
    show_best = per_symbol_best_df.copy().rename(columns={
        "symbol": "币对",
        "strategy": "策略名称",
        "avg_test_return_pct": "滚动样本外收益(%)",
        "avg_test_dd_pct": "滚动样本外回撤(%)",
        "avg_test_sharpe": "滚动样本外Sharpe",
        "full_return_pct": "全样本收益(%)",
        "full_dd_pct": "全样本回撤(%)",
        "full_sharpe": "全样本Sharpe",
        "score": "综合评分",
    })

    show_best = show_best[[
        "币对", "策略名称", "综合评分", "滚动样本外收益(%)", "滚动样本外回撤(%)",
        "滚动样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"
    ]].copy()

    for c in ["综合评分", "滚动样本外收益(%)", "滚动样本外回撤(%)", "滚动样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"]:
        show_best[c] = show_best[c].map(lambda x: safe_num(x, 2))

    lines.append(format_table(
        show_best,
        right_align_cols={"综合评分", "滚动样本外收益(%)", "滚动样本外回撤(%)", "滚动样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"}
    ))

    lines.append("")
    lines.append("【当前市场状态】")
    regime_rows = []
    for symbol in cfg["symbols"]:
        if symbol in snapshots:
            snap = snapshots[symbol]
            regime_rows.append({
                "币对": symbol,
                "当前价格": safe_num(snap.get("current_price"), 4),
                "当前 RSI7": safe_num(snap.get("current_rsi7"), 2),
                "当前 EMA20": safe_num(snap.get("current_ema20"), 4),
                "当前 VWAP": safe_num(snap.get("current_vwap"), 4),
                "最新市场状态": snap.get("latest_regime", "UNKNOWN")
            })
    regime_df = pd.DataFrame(regime_rows)
    lines.append(format_table(regime_df))

    lines.append("")
    lines.append("【输出文件】")
    lines.append(f"- 全部策略明细: {paths['detail_csv']}")
    lines.append(f"- 多币对综合排行榜: {paths['overall_csv']}")
    lines.append(f"- 各币对最佳策略: {paths['best_csv']}")
    lines.append(f"- 市场快照: {paths['snapshot_json']}")
    lines.append(f"- 综合文字总结: {paths['summary_txt']}")

    return "\n".join(lines)


def create_root_exports(paths):
    exported = {}

    root_summary = "latest_summary.txt"
    root_best_csv = "best_strategies.csv"
    root_overall_csv = "overall_ranking.csv"
    root_detail_csv = "all_strategy_details.csv"
    root_snapshot_json = "market_snapshots.json"
    root_zip = "backtest_package.zip"
    root_manifest = "generated_files.json"

    if os.path.exists(paths["summary_txt"]):
        shutil.copyfile(paths["summary_txt"], root_summary)
        exported["latest_summary"] = root_summary

    if os.path.exists(paths["best_csv"]):
        shutil.copyfile(paths["best_csv"], root_best_csv)
        exported["best_csv"] = root_best_csv

    if os.path.exists(paths["overall_csv"]):
        shutil.copyfile(paths["overall_csv"], root_overall_csv)
        exported["overall_csv"] = root_overall_csv

    if os.path.exists(paths["detail_csv"]):
        shutil.copyfile(paths["detail_csv"], root_detail_csv)
        exported["detail_csv"] = root_detail_csv

    if os.path.exists(paths["snapshot_json"]):
        shutil.copyfile(paths["snapshot_json"], root_snapshot_json)
        exported["snapshot_json"] = root_snapshot_json

    with zipfile.ZipFile(root_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, file_path in exported.items():
            if os.path.exists(file_path):
                zf.write(file_path, arcname=os.path.basename(file_path))
        for p in paths.values():
            if os.path.exists(p):
                zf.write(p, arcname=p)

    exported["zip_package"] = root_zip

    with open(root_manifest, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    exported["manifest"] = root_manifest
    return exported


# =========================================================
# 主程序
# =========================================================
def main():
    all_results = []
    snapshots = {}

    for symbol in CONFIG["symbols"]:
        one = backtest_one_symbol(symbol, CONFIG)
        if one is None:
            continue
        all_results.append(one["result_df"])
        snapshots[symbol] = one["snapshot"]

    if not all_results:
        fail_msg = "没有足够数据可以完成回测。"
        with open("latest_summary.txt", "w", encoding="utf-8") as f:
            f.write(fail_msg)
        print(fail_msg)
        return

    combined_detail_df, overall_df = aggregate_across_symbols(all_results)

    per_symbol_best_df = (
        combined_detail_df
        .sort_values(by=["symbol", "score", "avg_test_return_pct"], ascending=[True, False, False])
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    paths = save_reports(combined_detail_df, overall_df, per_symbol_best_df, snapshots)

    summary_text = build_summary_text(CONFIG, overall_df, per_symbol_best_df, snapshots, paths)

    with open(paths["summary_txt"], "w", encoding="utf-8") as f:
        f.write(summary_text)

    root_exports = create_root_exports(paths)

    print(summary_text)
    print("")
    print("【根目录导出文件】")
    for k, v in root_exports.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
