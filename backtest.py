import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unicodedata

VERSION = "V8.1_多币对_多策略_强化抓数版"
REPORT_DIR = "reports"

CONFIG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT"],
    "interval": "5m",
    "target_bars": 3000,          # 目标抓取K线数
    "request_limit": 1000,        # Bitget v2 candles 最大 1000
    "max_pages": 8,               # 最多翻页次数
    "fee_rate": 0.0006,
    "slippage_rate": 0.0002,
    "train_ratio": 0.6,
    "min_bars_required": 1200     # 至少要有这么多K线才回测
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


# =========================================================
# 数据抓取
# =========================================================
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


def fetch_candles_bitget(symbol="BTCUSDT", interval="5m", target_bars=3000, request_limit=1000, max_pages=8):
    """
    使用 Bitget v2 合约 candles 接口反复往前翻页抓更旧数据。
    """
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

            # 往前翻页：取当前最老K线之前
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

    df["ret"] = df["close"].pct_change().fillna(0)
    df["dist_ema20"] = (df["close"] / df["ema_20"]) - 1
    df["dist_ema50"] = (df["close"] / df["ema_50"]) - 1

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
# 策略生成
# =========================================================
def signal_from_conditions(long_cond, short_cond):
    return pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)))


def generate_strategies(df: pd.DataFrame):
    s = {}

    ema_pairs = [(9, 20), (12, 34), (20, 50), (21, 55), (50, 200)]
    for fast, slow in ema_pairs:
        s[f"EMA交叉_{fast}_{slow}"] = signal_from_conditions(
            df[f"ema_{fast}"] > df[f"ema_{slow}"],
            df[f"ema_{fast}"] < df[f"ema_{slow}"]
        )

    trend_cfgs = [
        (20, 50, 55, 45),
        (21, 55, 58, 42),
        (50, 200, 55, 45),
    ]
    for fast, slow, rsi_up, rsi_dn in trend_cfgs:
        s[f"趋势过滤_EMA{fast}_{slow}_RSI{rsi_up}_{rsi_dn}"] = signal_from_conditions(
            (df[f"ema_{fast}"] > df[f"ema_{slow}"]) & (df["rsi_14"] > rsi_up),
            (df[f"ema_{fast}"] < df[f"ema_{slow}"]) & (df["rsi_14"] < rsi_dn)
        )

    for up, dn in [(55, 45), (60, 40), (52, 48)]:
        s[f"MACD_RSI过滤_{up}_{dn}"] = signal_from_conditions(
            (df["macd"] > df["macd_signal"]) & (df["rsi_7"] > up),
            (df["macd"] < df["macd_signal"]) & (df["rsi_7"] < dn)
        )

    for ema in [20, 50]:
        s[f"MACD柱体_EMA{ema}"] = signal_from_conditions(
            (df["macd_hist"] > 0) & (df["close"] > df[f"ema_{ema}"]),
            (df["macd_hist"] < 0) & (df["close"] < df[f"ema_{ema}"])
        )

    for vol_len, mult in [(20, 1.2), (20, 1.4), (50, 1.2)]:
        vol_col = f"vol_ma_{vol_len}"
        s[f"放量突破_VOL{vol_len}_{mult}"] = signal_from_conditions(
            (df["close"] > df["ema_20"]) & (df["base_vol"] > df[vol_col] * mult),
            (df["close"] < df["ema_20"]) & (df["base_vol"] > df[vol_col] * mult)
        )

    for ema, mult in [(20, 0.8), (20, 1.0), (50, 0.8), (50, 1.0)]:
        s[f"ATR突破_EMA{ema}_{mult}"] = signal_from_conditions(
            df["close"] > (df[f"ema_{ema}"] + df["atr_14"] * mult),
            df["close"] < (df[f"ema_{ema}"] - df["atr_14"] * mult)
        )

    for low, high in [(20, 80), (25, 75), (30, 70)]:
        s[f"RSI反转_{low}_{high}"] = signal_from_conditions(
            df["rsi_7"] < low,
            df["rsi_7"] > high
        )

    for ema, dist in [(20, 0.015), (20, 0.02), (50, 0.02), (50, 0.03)]:
        dist_col = f"dist_ema{ema}"
        s[f"偏离EMA{ema}_反转_{dist}"] = signal_from_conditions(
            df[dist_col] < -dist,
            df[dist_col] > dist
        )

    for n in [20, 55]:
        s[f"Donchian突破_{n}"] = signal_from_conditions(
            df["close"] > df[f"donchian_high_{n}"],
            df["close"] < df[f"donchian_low_{n}"]
        )

    s["Bollinger反转_20_2.0"] = signal_from_conditions(
        df["close"] < df["bb_lower_20"],
        df["close"] > df["bb_upper_20"]
    )

    s["低位修复_RSI_MACD"] = signal_from_conditions(
        (df["rsi_7"] < 30) & (df["macd_hist"] > df["macd_hist"].shift(1)),
        (df["rsi_7"] > 70) & (df["macd_hist"] < df["macd_hist"].shift(1))
    )

    s["多因子共振_趋势版"] = signal_from_conditions(
        (df["close"] > df["ema_200"]) &
        (df["ema_20"] > df["ema_50"]) &
        (df["macd_hist"] > 0) &
        (df["rsi_14"] > 55) &
        (df["base_vol"] > df["vol_ma_20"]),
        (df["close"] < df["ema_200"]) &
        (df["ema_20"] < df["ema_50"]) &
        (df["macd_hist"] < 0) &
        (df["rsi_14"] < 45) &
        (df["base_vol"] > df["vol_ma_20"])
    )

    s["多因子共振_动能版"] = signal_from_conditions(
        (df["close"] > df["ema_50"]) &
        (df["ema_9"] > df["ema_20"]) &
        (df["macd"] > df["macd_signal"]) &
        (df["rsi_7"] > 60),
        (df["close"] < df["ema_50"]) &
        (df["ema_9"] < df["ema_20"]) &
        (df["macd"] < df["macd_signal"]) &
        (df["rsi_7"] < 40)
    )

    s["多因子共振_保守版"] = signal_from_conditions(
        (df["close"] > df["ema_200"]) &
        (df["rsi_14"] > 58) &
        (df["base_vol"] > df["vol_ma_20"] * 1.2),
        (df["close"] < df["ema_200"]) &
        (df["rsi_14"] < 42) &
        (df["base_vol"] > df["vol_ma_20"] * 1.2)
    )

    s["价格站上双均线_20_50"] = signal_from_conditions(
        (df["close"] > df["ema_20"]) & (df["close"] > df["ema_50"]),
        (df["close"] < df["ema_20"]) & (df["close"] < df["ema_50"])
    )

    s["价格站上双均线_50_200"] = signal_from_conditions(
        (df["close"] > df["ema_50"]) & (df["close"] > df["ema_200"]),
        (df["close"] < df["ema_50"]) & (df["close"] < df["ema_200"])
    )

    cleaned = {}
    for k, v in s.items():
        sig = pd.Series(v, index=df.index).fillna(0)
        if sig.abs().sum() > 0:
            cleaned[k] = sig

    return cleaned


# =========================================================
# 回测
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


def run_backtest(df: pd.DataFrame, signal: pd.Series, fee_rate=0.0006, slippage_rate=0.0002, annual_factor=72576):
    signal = pd.Series(signal, index=df.index).fillna(0)

    pos = signal.replace(0, np.nan).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)

    trades = pos.diff().fillna(0).abs()
    gross_ret = pos * df["ret"]

    trading_cost = trades * (fee_rate + slippage_rate)
    net_ret = gross_ret - trading_cost

    m = calc_equity_metrics(net_ret, annual_factor)
    exposure = (pos != 0).mean() * 100

    return {
        "净收益(%)": m["total_return"],
        "最大回撤(%)": m["max_dd"],
        "Sharpe": m["sharpe"],
        "胜率(%)": m["win_rate"],
        "交易次数": int((trades > 0).sum()),
        "信号变更次数": int(trades.sum()),
        "持仓占比(%)": exposure,
        "最终资金曲线": m["equity"]
    }


def run_train_test_backtest(df: pd.DataFrame, signal: pd.Series, interval: str, fee_rate: float, slippage_rate: float, train_ratio: float):
    annual_factor = annual_factor_from_interval(interval)
    n = len(df)
    split_idx = int(n * train_ratio)

    if split_idx < 300 or (n - split_idx) < 300:
        return None

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    sig_train = signal.iloc[:split_idx]
    sig_test = signal.iloc[split_idx:]

    train_res = run_backtest(df_train, sig_train, fee_rate, slippage_rate, annual_factor)
    test_res = run_backtest(df_test, sig_test, fee_rate, slippage_rate, annual_factor)
    full_res = run_backtest(df, signal, fee_rate, slippage_rate, annual_factor)

    return {
        "train": train_res,
        "test": test_res,
        "full": full_res
    }


def compute_strategy_score(test_return, test_dd, test_sharpe, win_rate, trades, consistency_bonus):
    score = 0.0
    score += test_return * 1.8
    score += test_sharpe * 8.0
    score -= abs(test_dd) * 1.5
    score += (win_rate - 50) * 0.4
    score += consistency_bonus

    if trades < 3:
        score -= 18
    elif trades < 8:
        score -= 8
    elif trades > 120:
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
    strategies = generate_strategies(df)

    rows = []

    for strategy_name, signal in strategies.items():
        res = run_train_test_backtest(
            df=df,
            signal=signal,
            interval=cfg["interval"],
            fee_rate=cfg["fee_rate"],
            slippage_rate=cfg["slippage_rate"],
            train_ratio=cfg["train_ratio"]
        )
        if res is None:
            continue

        train_r = res["train"]
        test_r = res["test"]
        full_r = res["full"]

        consistency_bonus = 0
        if test_r["净收益(%)"] > 0:
            consistency_bonus += 5
        if test_r["Sharpe"] > 0:
            consistency_bonus += 3
        if abs(test_r["最大回撤(%)"]) < 10:
            consistency_bonus += 3

        score = compute_strategy_score(
            test_return=test_r["净收益(%)"],
            test_dd=test_r["最大回撤(%)"],
            test_sharpe=test_r["Sharpe"],
            win_rate=test_r["胜率(%)"],
            trades=test_r["交易次数"],
            consistency_bonus=consistency_bonus
        )

        rows.append({
            "symbol": symbol,
            "strategy": strategy_name,
            "bars": len(df),
            "train_return_pct": train_r["净收益(%)"],
            "train_dd_pct": train_r["最大回撤(%)"],
            "train_sharpe": train_r["Sharpe"],
            "test_return_pct": test_r["净收益(%)"],
            "test_dd_pct": test_r["最大回撤(%)"],
            "test_sharpe": test_r["Sharpe"],
            "test_win_rate_pct": test_r["胜率(%)"],
            "test_trades": test_r["交易次数"],
            "full_return_pct": full_r["净收益(%)"],
            "full_dd_pct": full_r["最大回撤(%)"],
            "full_sharpe": full_r["Sharpe"],
            "full_win_rate_pct": full_r["胜率(%)"],
            "full_trades": full_r["交易次数"],
            "score": score
        })

    if not rows:
        return None

    result_df = pd.DataFrame(rows).sort_values(by=["score", "test_return_pct"], ascending=[False, False]).reset_index(drop=True)

    snapshot = {
        "symbol": symbol,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": float(df["close"].iloc[-1]),
        "current_ema20": float(df["ema_20"].iloc[-1]),
        "current_macd": float(df["macd"].iloc[-1]),
        "current_rsi7": float(df["rsi_7"].iloc[-1]),
        "bars": len(df),
        "interval": cfg["interval"]
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
        avg_test_return_pct=("test_return_pct", "mean"),
        median_test_return_pct=("test_return_pct", "median"),
        avg_test_dd_pct=("test_dd_pct", "mean"),
        avg_test_sharpe=("test_sharpe", "mean"),
        avg_test_win_rate_pct=("test_win_rate_pct", "mean"),
        avg_full_return_pct=("full_return_pct", "mean"),
        avg_full_dd_pct=("full_dd_pct", "mean"),
        avg_full_sharpe=("full_sharpe", "mean"),
        positive_test_symbols=("test_return_pct", lambda x: int((x > 0).sum())),
        positive_sharpe_symbols=("test_sharpe", lambda x: int((x > 0).sum())),
    ).reset_index()

    grouped["robustness_score"] = (
        grouped["avg_score"] * 0.9
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


def build_summary_text(cfg, overall_df, per_symbol_best_df, paths):
    lines = []
    lines.append(f"[{VERSION}]")
    lines.append(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"回测币对: {', '.join(cfg['symbols'])}")
    lines.append(f"周期: {cfg['interval']}")
    lines.append(f"目标K线数: {cfg['target_bars']}")
    lines.append(f"单次请求上限: {cfg['request_limit']}")
    lines.append(f"最大翻页数: {cfg['max_pages']}")
    lines.append(f"手续费: {cfg['fee_rate']}")
    lines.append(f"滑点: {cfg['slippage_rate']}")
    lines.append(f"样本内比例: {cfg['train_ratio']}")
    lines.append("=" * 130)

    lines.append("【多币对综合最佳策略 Top 15】")
    show_overall = overall_df.head(15).copy().rename(columns={
        "strategy": "策略名称",
        "symbols_tested": "测试币数",
        "robustness_score": "综合鲁棒评分",
        "avg_test_return_pct": "平均样本外收益(%)",
        "avg_test_dd_pct": "平均样本外回撤(%)",
        "avg_test_sharpe": "平均样本外Sharpe",
        "positive_test_symbols": "样本外盈利币数",
        "positive_sharpe_symbols": "正Sharpe币数",
    })

    show_overall = show_overall[[
        "策略名称", "测试币数", "综合鲁棒评分", "平均样本外收益(%)",
        "平均样本外回撤(%)", "平均样本外Sharpe", "样本外盈利币数", "正Sharpe币数"
    ]].copy()

    for c in ["综合鲁棒评分", "平均样本外收益(%)", "平均样本外回撤(%)", "平均样本外Sharpe"]:
        show_overall[c] = show_overall[c].map(lambda x: safe_num(x, 2))

    lines.append(format_table(
        show_overall,
        right_align_cols={"测试币数", "综合鲁棒评分", "平均样本外收益(%)", "平均样本外回撤(%)", "平均样本外Sharpe", "样本外盈利币数", "正Sharpe币数"}
    ))

    lines.append("")
    lines.append("【各币对最佳策略】")
    show_best = per_symbol_best_df.copy().rename(columns={
        "symbol": "币对",
        "strategy": "策略名称",
        "test_return_pct": "样本外收益(%)",
        "test_dd_pct": "样本外回撤(%)",
        "test_sharpe": "样本外Sharpe",
        "full_return_pct": "全样本收益(%)",
        "full_dd_pct": "全样本回撤(%)",
        "full_sharpe": "全样本Sharpe",
        "score": "综合评分",
    })

    show_best = show_best[[
        "币对", "策略名称", "综合评分", "样本外收益(%)", "样本外回撤(%)",
        "样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"
    ]].copy()

    for c in ["综合评分", "样本外收益(%)", "样本外回撤(%)", "样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"]:
        show_best[c] = show_best[c].map(lambda x: safe_num(x, 2))

    lines.append(format_table(
        show_best,
        right_align_cols={"综合评分", "样本外收益(%)", "样本外回撤(%)", "样本外Sharpe", "全样本收益(%)", "全样本回撤(%)", "全样本Sharpe"}
    ))

    lines.append("")
    lines.append("【输出文件】")
    lines.append(f"- 全部策略明细: {paths['detail_csv']}")
    lines.append(f"- 多币对综合排行榜: {paths['overall_csv']}")
    lines.append(f"- 各币对最佳策略: {paths['best_csv']}")
    lines.append(f"- 市场快照: {paths['snapshot_json']}")
    lines.append(f"- 综合文字总结: {paths['summary_txt']}")

    return "\n".join(lines)


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
        print("没有足够数据可以完成回测。")
        return

    combined_detail_df, overall_df = aggregate_across_symbols(all_results)

    per_symbol_best_df = (
        combined_detail_df
        .sort_values(by=["symbol", "score", "test_return_pct"], ascending=[True, False, False])
        .groupby("symbol", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    paths = save_reports(combined_detail_df, overall_df, per_symbol_best_df, snapshots)

    summary_text = build_summary_text(CONFIG, overall_df, per_symbol_best_df, paths)

    with open(paths["summary_txt"], "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open("latest_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
