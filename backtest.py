import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

VERSION = "V6.5_候选池验证版"
BASE_URL = "https://api.bitget.com/api/v2/mix/market/candles"


# =========================
# 基础工具
# =========================
def 当前时间字符串():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def 创建目录(path: str):
    os.makedirs(path, exist_ok=True)


def 百分比格式(x):
    return f"{x:.2f}%"


def 数字格式(x):
    return f"{x:.2f}"


def 纯ASCII表格(df: pd.DataFrame) -> str:
    if df.empty:
        return "(空表)"

    cols = list(df.columns)
    rows = [[str(x) for x in row] for row in df.values.tolist()]
    widths = []

    for i, col in enumerate(cols):
        max_len = len(str(col))
        for row in rows:
            max_len = max(max_len, len(row[i]))
        widths.append(max_len)

    def fmt_row(row):
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols)))

    sep = "-+-".join("-" * w for w in widths)
    output = [fmt_row(cols), sep]
    for row in rows:
        output.append(fmt_row(row))
    return "\n".join(output)


# =========================
# 数据抓取
# =========================
def 周期毫秒(interval: str) -> int:
    mapping = {
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1H": 60 * 60 * 1000,
        "4H": 4 * 60 * 60 * 1000,
    }
    return mapping[interval]


def 周期年化bar数(interval: str) -> int:
    mapping = {
        "3m": 20 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "30m": 2 * 24 * 365,
        "1H": 24 * 365,
        "4H": 6 * 365,
    }
    return mapping[interval]


def 抓取指定bar数量(symbol="BTCUSDT", interval="5m", target_bars=9000, sleep_sec=0.12):
    print(f"[{VERSION}] 抓取 {symbol} {interval}，目标 bars={target_bars}")

    end_time = str(int(time.time() * 1000))
    all_data = []
    seen_end_times = set()

    # 每次最多 1000，给一点冗余空间
    max_rounds = max(3, int(target_bars / 800) + 5)

    for _ in range(max_rounds):
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "endTime": end_time,
            "limit": "1000",
        }

        try:
            res = requests.get(BASE_URL, params=params, timeout=10)
            res.raise_for_status()
            payload = res.json()
            data = payload.get("data", [])

            if not data:
                print(f"{symbol} {interval} 没有更多数据，停止抓取。")
                break

            if end_time in seen_end_times:
                print(f"{symbol} {interval} endTime 重复，停止抓取。")
                break
            seen_end_times.add(end_time)

            all_data.extend(data)

            if len(all_data) >= target_bars + 500:
                break

            new_end_time = str(data[-1][0])
            if new_end_time == end_time:
                print(f"{symbol} {interval} endTime 未变化，停止抓取。")
                break

            end_time = new_end_time
            time.sleep(sleep_sec)

        except requests.RequestException as e:
            print(f"{symbol} {interval} 请求失败: {e}")
            break
        except (ValueError, KeyError, TypeError) as e:
            print(f"{symbol} {interval} 数据解析失败: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "base_vol",
            "quote_vol",
        ],
    )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    if len(df) > target_bars:
        df = df.tail(target_bars).reset_index(drop=True)

    print(f"{symbol} {interval} 实际获取 {len(df)} 根K")
    return df


def 切取最近bars(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if len(df) <= bars:
        return df.copy().reset_index(drop=True)
    return df.tail(bars).reset_index(drop=True)


# =========================
# 指标
# =========================
def 计算指标(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for span in [10, 20, 50, 100, 200]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    df["vol_ma_20"] = df["base_vol"].rolling(20).mean()
    df["vol_ma_50"] = df["base_vol"].rolling(50).mean()

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for rsi_period in [7, 10, 14, 21]:
        avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{rsi_period}"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper_2"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower_2"] = df["bb_mid"] - 2 * df["bb_std"]

    df["ret"] = df["close"].pct_change().fillna(0)

    return df


def 条件转信号(long_cond, short_cond):
    return np.where(long_cond, 1, np.where(short_cond, -1, 0))


# =========================
# 全量策略库
# 只为了复用旧策略ID，方便你从上一轮延续
# =========================
def 生成策略库(df: pd.DataFrame):
    strategies = {}
    catalog = []
    sid = 1

    def 新增策略(name_cn: str, family_cn: str, params: dict, signal):
        nonlocal sid
        strategy_id = f"S{sid:03d}"
        sid += 1
        strategies[strategy_id] = {
            "name_cn": name_cn,
            "family_cn": family_cn,
            "params": params,
            "signal": signal,
        }
        catalog.append(
            {
                "策略ID": strategy_id,
                "策略名称": name_cn,
                "策略分类": family_cn,
                "参数": str(params),
            }
        )

    趋势多 = {
        "ema20上穿ema50": df["ema_20"] > df["ema_50"],
        "ema20上穿ema200": df["ema_20"] > df["ema_200"],
        "收盘站上ema200": df["close"] > df["ema_200"],
    }
    趋势空 = {
        "ema20上穿ema50": df["ema_20"] < df["ema_50"],
        "ema20上穿ema200": df["ema_20"] < df["ema_200"],
        "收盘站上ema200": df["close"] < df["ema_200"],
    }

    rsi_pairs = [(55, 45), (60, 40), (65, 35)]
    macd_modes = {
        "MACD金叉死叉": (df["macd"] > df["macd_signal"], df["macd"] < df["macd_signal"]),
        "MACD零轴上下": (df["macd"] > 0, df["macd"] < 0),
        "MACD柱体正负": (df["macd_hist"] > 0, df["macd_hist"] < 0),
    }
    volume_modes = {
        "不加成交量过滤": (pd.Series(True, index=df.index), pd.Series(True, index=df.index)),
        "成交量大于20均量": (df["base_vol"] > df["vol_ma_20"], df["base_vol"] > df["vol_ma_20"]),
        "成交量大于1.5倍20均量": (df["base_vol"] > df["vol_ma_20"] * 1.5, df["base_vol"] > df["vol_ma_20"] * 1.5),
    }

    for trend_key in 趋势多.keys():
        for (rsi_hi, rsi_lo) in rsi_pairs:
            for macd_key in macd_modes.keys():
                for vol_key in volume_modes.keys():
                    long_cond = (
                        趋势多[trend_key]
                        & (df["rsi_14"] > rsi_hi)
                        & macd_modes[macd_key][0]
                        & volume_modes[vol_key][0]
                    )
                    short_cond = (
                        趋势空[trend_key]
                        & (df["rsi_14"] < rsi_lo)
                        & macd_modes[macd_key][1]
                        & volume_modes[vol_key][1]
                    )
                    新增策略(
                        name_cn=f"趋势确认 {trend_key} RSI{rsi_hi}/{rsi_lo} {macd_key} {vol_key}",
                        family_cn="趋势确认类",
                        params={
                            "趋势条件": trend_key,
                            "RSI高": rsi_hi,
                            "RSI低": rsi_lo,
                            "MACD模式": macd_key,
                            "成交量模式": vol_key,
                        },
                        signal=条件转信号(long_cond, short_cond),
                    )

    for atr_mult in [0.5, 1.0, 1.5]:
        for trend_key in ["ema20上穿ema50", "ema20上穿ema200"]:
            for use_vol in [False, True]:
                vol_cond = df["base_vol"] > df["vol_ma_20"] if use_vol else pd.Series(True, index=df.index)
                long_cond = (
                    (df["close"] > (df["ema_20"] + atr_mult * df["atr_14"]))
                    & 趋势多[trend_key]
                    & vol_cond
                )
                short_cond = (
                    (df["close"] < (df["ema_20"] - atr_mult * df["atr_14"]))
                    & 趋势空[trend_key]
                    & vol_cond
                )
                新增策略(
                    name_cn=f"ATR突破 {trend_key} 倍数{atr_mult} 成交量过滤={use_vol}",
                    family_cn="ATR突破类",
                    params={"趋势条件": trend_key, "ATR倍数": atr_mult, "成交量过滤": use_vol},
                    signal=条件转信号(long_cond, short_cond),
                )

    for rsi_period in [7, 10, 14, 21]:
        for low in [18, 20, 22, 25, 30]:
            for high in [70, 75, 78, 80, 82]:
                for with_trend in [False, True]:
                    if low >= high:
                        continue
                    rsi_col = f"rsi_{rsi_period}"
                    base_long = df[rsi_col] < low
                    base_short = df[rsi_col] > high

                    if with_trend:
                        long_cond = base_long & (df["close"] > df["ema_200"])
                        short_cond = base_short & (df["close"] < df["ema_200"])
                    else:
                        long_cond = base_long
                        short_cond = base_short

                    新增策略(
                        name_cn=f"RSI均值回归 RSI{rsi_period} {low}/{high} 趋势过滤={with_trend}",
                        family_cn="RSI均值回归类",
                        params={
                            "RSI周期": rsi_period,
                            "RSI低点": low,
                            "RSI高点": high,
                            "趋势过滤": with_trend,
                        },
                        signal=条件转信号(long_cond, short_cond),
                    )

    for with_trend in [False, True]:
        long_cond = df["close"] < df["bb_lower_2"]
        short_cond = df["close"] > df["bb_upper_2"]
        if with_trend:
            long_cond = long_cond & (df["close"] > df["ema_200"])
            short_cond = short_cond & (df["close"] < df["ema_200"])

        新增策略(
            name_cn=f"布林带回归 趋势过滤={with_trend}",
            family_cn="布林回归类",
            params={"趋势过滤": with_trend},
            signal=条件转信号(long_cond, short_cond),
        )

    for pct in [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020]:
        for base in ["ema_20", "ema_50"]:
            long_cond = df["close"] < df[base] * (1 - pct)
            short_cond = df["close"] > df[base] * (1 + pct)
            新增策略(
                name_cn=f"均线偏离 {base} 偏离{pct:.3f}",
                family_cn="均线偏离类",
                params={"基础均线": base, "偏离比例": pct},
                signal=条件转信号(long_cond, short_cond),
            )

    hybrid_configs = [
        ("收盘站上ema200", "MACD金叉死叉", "成交量大于20均量", 55, 45),
        ("ema20上穿ema50", "MACD柱体正负", "成交量大于20均量", 60, 40),
        ("ema20上穿ema200", "MACD零轴上下", "成交量大于1.5倍20均量", 55, 45),
        ("收盘站上ema200", "MACD柱体正负", "不加成交量过滤", 65, 35),
    ]
    for trend_key, macd_key, vol_key, rsi_hi, rsi_lo in hybrid_configs:
        long_cond = (
            趋势多[trend_key]
            & macd_modes[macd_key][0]
            & volume_modes[vol_key][0]
            & (df["rsi_14"] > rsi_hi)
            & (df["close"] > df["ema_20"])
        )
        short_cond = (
            趋势空[trend_key]
            & macd_modes[macd_key][1]
            & volume_modes[vol_key][1]
            & (df["rsi_14"] < rsi_lo)
            & (df["close"] < df["ema_20"])
        )
        新增策略(
            name_cn=f"混合共振 {trend_key} {macd_key} {vol_key} RSI{rsi_hi}/{rsi_lo}",
            family_cn="混合共振类",
            params={
                "趋势条件": trend_key,
                "MACD模式": macd_key,
                "成交量模式": vol_key,
                "RSI高": rsi_hi,
                "RSI低": rsi_lo,
            },
            signal=条件转信号(long_cond, short_cond),
        )

    return strategies, pd.DataFrame(catalog)


# =========================
# 候选池执行版本
# =========================
def 生成执行版本():
    return [
        {
            "执行版本": "原版",
            "固定止损": None,
            "固定止盈": None,
            "ATR止损倍数": None,
            "ATR止盈倍数": None,
            "冷却K数": 0,
            "最大持仓K数": None,
        },
        {
            "执行版本": "固定止损止盈_1.5_3.0",
            "固定止损": 0.015,
            "固定止盈": 0.030,
            "ATR止损倍数": None,
            "ATR止盈倍数": None,
            "冷却K数": 0,
            "最大持仓K数": None,
        },
        {
            "执行版本": "ATR止损止盈_1.2_2.0",
            "固定止损": None,
            "固定止盈": None,
            "ATR止损倍数": 1.2,
            "ATR止盈倍数": 2.0,
            "冷却K数": 0,
            "最大持仓K数": None,
        },
        {
            "执行版本": "冷却12K",
            "固定止损": None,
            "固定止盈": None,
            "ATR止损倍数": None,
            "ATR止盈倍数": None,
            "冷却K数": 12,
            "最大持仓K数": None,
        },
        {
            "执行版本": "最大持仓48K",
            "固定止损": None,
            "固定止盈": None,
            "ATR止损倍数": None,
            "ATR止盈倍数": None,
            "冷却K数": 0,
            "最大持仓K数": 48,
        },
    ]


# =========================
# 回测核心
# =========================
def 最大回撤(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return float(drawdown.min())


def 夏普(net_ret: pd.Series, bars_per_year: float) -> float:
    std = net_ret.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((net_ret.mean() / std) * np.sqrt(bars_per_year))


def 卡玛(total_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return float(total_return / abs(max_dd))


def 回测单策略_带风控(
    df: pd.DataFrame,
    raw_signal,
    interval="5m",
    fee_rate=0.0006,
    slippage=0.0002,
    固定止损=None,
    固定止盈=None,
    ATR止损倍数=None,
    ATR止盈倍数=None,
    冷却K数=0,
    最大持仓K数=None,
    策略ID="",
    策略名称="",
    执行版本="原版",
):
    position = 0
    entry_price = None
    entry_time = None
    entry_bar = None
    cooldown_until = -1

    positions = []
    net_rets = []
    trade_records = []

    prev_pos = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["close"]
        high = row["high"]
        low = row["low"]
        atr = row["atr_14"]
        signal = raw_signal[i]

        exit_reason = None
        trade_cost = 0.0

        # 先处理已有仓位的退出逻辑
        if position != 0 and entry_price is not None:
            持仓K数 = i - entry_bar if entry_bar is not None else 0

            # 固定止损止盈
            if 固定止损 is not None or 固定止盈 is not None:
                if position == 1:
                    if 固定止损 is not None and low <= entry_price * (1 - 固定止损):
                        exit_reason = f"固定止损_{固定止损}"
                    elif 固定止盈 is not None and high >= entry_price * (1 + 固定止盈):
                        exit_reason = f"固定止盈_{固定止盈}"
                elif position == -1:
                    if 固定止损 is not None and high >= entry_price * (1 + 固定止损):
                        exit_reason = f"固定止损_{固定止损}"
                    elif 固定止盈 is not None and low <= entry_price * (1 - 固定止盈):
                        exit_reason = f"固定止盈_{固定止盈}"

            # ATR止损止盈
            if exit_reason is None and (ATR止损倍数 is not None or ATR止盈倍数 is not None) and pd.notna(atr):
                if position == 1:
                    if ATR止损倍数 is not None and low <= entry_price - atr * ATR止损倍数:
                        exit_reason = f"ATR止损_{ATR止损倍数}"
                    elif ATR止盈倍数 is not None and high >= entry_price + atr * ATR止盈倍数:
                        exit_reason = f"ATR止盈_{ATR止盈倍数}"
                elif position == -1:
                    if ATR止损倍数 is not None and high >= entry_price + atr * ATR止损倍数:
                        exit_reason = f"ATR止损_{ATR止损倍数}"
                    elif ATR止盈倍数 is not None and low <= entry_price - atr * ATR止盈倍数:
                        exit_reason = f"ATR止盈_{ATR止盈倍数}"

            # 最大持仓时长
            if exit_reason is None and 最大持仓K数 is not None and 持仓K数 >= 最大持仓K数:
                exit_reason = f"最大持仓{最大持仓K数}K"

            # 反向信号退出
            if exit_reason is None:
                if position == 1 and signal == -1:
                    exit_reason = "反向信号"
                elif position == -1 and signal == 1:
                    exit_reason = "反向信号"

        # 有退出
        if exit_reason is not None and position != 0:
            exit_price = price
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if position == 1 else ((entry_price - exit_price) / entry_price) * 100

            trade_records.append(
                {
                    "策略ID": 策略ID,
                    "策略名称": 策略名称,
                    "执行版本": 执行版本,
                    "进场时间": entry_time,
                    "出场时间": row["timestamp"],
                    "方向": "做多" if position == 1 else "做空",
                    "进场价格": entry_price,
                    "出场价格": exit_price,
                    "持仓K数": i - entry_bar if entry_bar is not None else None,
                    "出场原因": exit_reason,
                    "单笔盈亏(%)": pnl_pct,
                }
            )

            trade_cost += (fee_rate + slippage)
            position = 0
            entry_price = None
            entry_time = None
            entry_bar = None
            cooldown_until = i + 冷却K数 if 冷却K数 > 0 else i

        # 再决定是否开新仓
        if position == 0 and i >= cooldown_until:
            if signal == 1:
                position = 1
                entry_price = price
                entry_time = row["timestamp"]
                entry_bar = i
                trade_cost += (fee_rate + slippage)
            elif signal == -1:
                position = -1
                entry_price = price
                entry_time = row["timestamp"]
                entry_bar = i
                trade_cost += (fee_rate + slippage)

        # 计算当根收益
        ret = row["ret"]
        gross_ret = position * ret
        net_ret = gross_ret - trade_cost

        positions.append(position)
        net_rets.append(net_ret)
        prev_pos = position

    pos_series = pd.Series(positions, index=df.index)
    net_ret_series = pd.Series(net_rets, index=df.index)
    equity_curve = (1 + net_ret_series).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1)
    max_dd = 最大回撤(equity_curve)
    sharpe = 夏普(net_ret_series, 周期年化bar数(interval))
    calmar = 卡玛(total_return, max_dd)

    entries = len([x for x in trade_records if x["进场时间"] is not None])
    flips = int(((pos_series * pos_series.shift(1).fillna(0)) < 0).sum())
    exposure = float((pos_series != 0).mean())

    non_zero_ret = net_ret_series[net_ret_series != 0]
    win_rate = float((non_zero_ret > 0).mean()) if len(non_zero_ret) > 0 else 0.0

    avg_win = net_ret_series[net_ret_series > 0].mean()
    avg_loss = net_ret_series[net_ret_series < 0].mean()
    pnl_ratio = (
        float(abs(avg_win / avg_loss))
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
        else 0.0
    )

    return {
        "position": pos_series,
        "net_ret": net_ret_series,
        "equity_curve": equity_curve,
        "trade_records": pd.DataFrame(trade_records),
        "summary": {
            "收益率(%)": total_return * 100,
            "最大回撤(%)": max_dd * 100,
            "夏普值": sharpe,
            "卡玛值": calmar,
            "开仓次数": entries,
            "反手次数": flips,
            "持仓占比(%)": exposure * 100,
            "胜率(%)": win_rate * 100,
            "盈亏比": pnl_ratio,
        },
    }


def 计算综合评分(report: pd.DataFrame) -> pd.DataFrame:
    df = report.copy()
    df["综合评分"] = (
        df["收益率(%)"] * 0.30
        + df["夏普值"] * 1.2
        + df["卡玛值"] * 5.0
        + df["胜率(%)"] * 0.02
        + df["盈亏比"] * 3.0
        - df["最大回撤(%)"].abs() * 0.45
    )

    df["是否通过基础门槛"] = (
        (df["开仓次数"] >= 5)
        & (df["持仓占比(%)"] <= 90)
        & (df["最大回撤(%)"] >= -15)
    )
    return df


# =========================
# 场景运行
# =========================
def 跑候选池场景(
    df_scene: pd.DataFrame,
    interval: str,
    候选策略ID列表: List[str],
    fee_rate=0.0006,
    slippage=0.0002,
):
    df_scene = 计算指标(df_scene)

    all_strategies, catalog_df = 生成策略库(df_scene)
    执行版本列表 = 生成执行版本()

    # 候选池过滤
    候选策略 = {}
    for sid in 候选策略ID列表:
        if sid in all_strategies:
            候选策略[sid] = all_strategies[sid]

    if not 候选策略:
        raise ValueError("候选池为空，请检查策略ID。")

    results = []
    trade_details = []

    for 策略ID, meta in 候选策略.items():
        raw_signal = meta["signal"]

        for cfg in 执行版本列表:
            result = 回测单策略_带风控(
                df=df_scene,
                raw_signal=raw_signal,
                interval=interval,
                fee_rate=fee_rate,
                slippage=slippage,
                固定止损=cfg["固定止损"],
                固定止盈=cfg["固定止盈"],
                ATR止损倍数=cfg["ATR止损倍数"],
                ATR止盈倍数=cfg["ATR止盈倍数"],
                冷却K数=cfg["冷却K数"],
                最大持仓K数=cfg["最大持仓K数"],
                策略ID=策略ID,
                策略名称=meta["name_cn"],
                执行版本=cfg["执行版本"],
            )

            summary = result["summary"]
            summary["策略ID"] = 策略ID
            summary["策略名称"] = meta["name_cn"]
            summary["策略分类"] = meta["family_cn"]
            summary["执行版本"] = cfg["执行版本"]
            results.append(summary)

            trade_df = result["trade_records"]
            if not trade_df.empty:
                trade_details.append(trade_df)

    report = pd.DataFrame(results)
    report = 计算综合评分(report)
    report = report.sort_values(
        by=["是否通过基础门槛", "综合评分"],
        ascending=[False, False]
    ).reset_index(drop=True)

    benchmark = ((1 + df_scene["ret"]).cumprod().iloc[-1] - 1) * 100

    if trade_details:
        trade_details_df = pd.concat(trade_details, ignore_index=True)
    else:
        trade_details_df = pd.DataFrame(columns=[
            "策略ID", "策略名称", "执行版本", "进场时间", "出场时间",
            "方向", "进场价格", "出场价格", "持仓K数", "出场原因", "单笔盈亏(%)"
        ])

    候选目录表 = catalog_df[catalog_df["策略ID"].isin(候选策略ID列表)].copy().reset_index(drop=True)

    return report, benchmark, 候选目录表, trade_details_df


# =========================
# 汇总
# =========================
def 汇总版本稳定度(所有场景结果: pd.DataFrame) -> pd.DataFrame:
    df = 所有场景结果.copy()

    df["进前10"] = df["场景内排名"] <= 10
    df["进前3"] = df["场景内排名"] <= 3
    df["冠军"] = df["场景内排名"] == 1

    grouped = df.groupby(
        ["策略ID", "策略名称", "策略分类", "执行版本"],
        as_index=False
    ).agg(
        场景数=("场景名", "count"),
        平均排名=("场景内排名", "mean"),
        前十次数=("进前10", "sum"),
        前三次数=("进前3", "sum"),
        冠军次数=("冠军", "sum"),
        平均收益率=("收益率(%)", "mean"),
        平均最大回撤=("最大回撤(%)", "mean"),
        平均夏普=("夏普值", "mean"),
        平均卡玛=("卡玛值", "mean"),
        平均胜率=("胜率(%)", "mean"),
        平均盈亏比=("盈亏比", "mean"),
        平均持仓占比=("持仓占比(%)", "mean"),
        平均开仓次数=("开仓次数", "mean"),
        平均评分=("综合评分", "mean"),
        通过门槛次数=("是否通过基础门槛", "sum"),
    )

    grouped["通过率(%)"] = grouped["通过门槛次数"] / grouped["场景数"] * 100
    grouped["前十率(%)"] = grouped["前十次数"] / grouped["场景数"] * 100
    grouped["前三率(%)"] = grouped["前三次数"] / grouped["场景数"] * 100
    grouped["冠军率(%)"] = grouped["冠军次数"] / grouped["场景数"] * 100

    grouped = grouped.sort_values(
        by=["通过门槛次数", "前三次数", "前十次数", "平均评分"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return grouped


def 构建控制台报表(report: pd.DataFrame, top_n=15):
    view = report.head(top_n).copy()
    view = view[
        [
            "策略ID",
            "执行版本",
            "收益率(%)",
            "最大回撤(%)",
            "夏普值",
            "卡玛值",
            "开仓次数",
            "持仓占比(%)",
            "胜率(%)",
            "盈亏比",
            "综合评分",
            "是否通过基础门槛",
        ]
    ]

    for col in ["收益率(%)", "最大回撤(%)", "持仓占比(%)", "胜率(%)"]:
        view[col] = view[col].map(百分比格式)

    for col in ["夏普值", "卡玛值", "盈亏比", "综合评分"]:
        view[col] = view[col].map(数字格式)

    view["是否通过基础门槛"] = view["是否通过基础门槛"].map(lambda x: "是" if x else "否")
    return view


def 构建全局控制台报表(稳定度表: pd.DataFrame, top_n=20):
    view = 稳定度表.head(top_n).copy()
    view = view[
        [
            "策略ID",
            "执行版本",
            "场景数",
            "平均排名",
            "前三次数",
            "前十次数",
            "冠军次数",
            "平均收益率",
            "平均最大回撤",
            "平均持仓占比",
            "平均开仓次数",
            "平均评分",
            "通过率(%)",
        ]
    ]

    for col in ["平均排名", "平均收益率", "平均最大回撤", "平均持仓占比", "平均开仓次数", "平均评分"]:
        view[col] = view[col].map(数字格式)

    view["通过率(%)"] = view["通过率(%)"].map(百分比格式)
    return view


# =========================
# 输出
# =========================
def 保存输出(
    场景摘要表: pd.DataFrame,
    所有场景结果: pd.DataFrame,
    版本稳定度表: pd.DataFrame,
    候选目录表: pd.DataFrame,
    全部交易明细: pd.DataFrame,
    场景冠军表: pd.DataFrame,
    全局控制台报表: pd.DataFrame,
):
    ts = 当前时间字符串()
    out_dir = os.path.join("outputs", ts)
    创建目录(out_dir)

    paths = {
        "输出目录": out_dir,
        "场景摘要文件": os.path.join(out_dir, f"场景摘要_{ts}.csv"),
        "全部结果文件": os.path.join(out_dir, f"候选池全部场景结果_{ts}.csv"),
        "版本稳定度文件": os.path.join(out_dir, f"候选池版本稳定度统计_{ts}.csv"),
        "候选目录文件": os.path.join(out_dir, f"候选池策略目录_{ts}.csv"),
        "交易明细文件": os.path.join(out_dir, f"逐笔交易明细_{ts}.csv"),
        "场景冠军文件": os.path.join(out_dir, f"场景冠军列表_{ts}.csv"),
        "摘要TXT文件": os.path.join(out_dir, f"总摘要_{ts}.txt"),
        "摘要MD文件": os.path.join(out_dir, f"总摘要_{ts}.md"),
    }

    场景摘要表.to_csv(paths["场景摘要文件"], index=False, encoding="utf-8-sig")
    所有场景结果.to_csv(paths["全部结果文件"], index=False, encoding="utf-8-sig")
    版本稳定度表.to_csv(paths["版本稳定度文件"], index=False, encoding="utf-8-sig")
    候选目录表.to_csv(paths["候选目录文件"], index=False, encoding="utf-8-sig")
    全部交易明细.to_csv(paths["交易明细文件"], index=False, encoding="utf-8-sig")
    场景冠军表.to_csv(paths["场景冠军文件"], index=False, encoding="utf-8-sig")

    with open(paths["摘要TXT文件"], "w", encoding="utf-8-sig") as f:
        f.write(f"[{VERSION}] 候选池验证总摘要\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"场景数: {len(场景摘要表)}\n")
        f.write(f"候选策略数: {候选目录表['策略ID'].nunique()}\n")
        f.write(f"执行版本数: {版本稳定度表['执行版本'].nunique() if not 版本稳定度表.empty else 0}\n\n")
        f.write("全局版本稳定度 Top 20\n")
        f.write(纯ASCII表格(全局控制台报表))
        f.write("\n")

    with open(paths["摘要MD文件"], "w", encoding="utf-8-sig") as f:
        f.write(f"# {VERSION} 候选池验证总摘要\n\n")
        f.write(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 场景数: {len(场景摘要表)}\n")
        f.write(f"- 候选策略数: {候选目录表['策略ID'].nunique()}\n\n")
        f.write("## 全局版本稳定度 Top 20\n\n")
        try:
            f.write(全局控制台报表.to_markdown(index=False))
        except Exception:
            f.write(全局控制台报表.to_string(index=False))
        f.write("\n")

    return paths


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    # 这里先放你目前最值得观察的一组示例
    # 你后面可以根据我的分析继续替换
    候选策略ID列表 = [
        "S178",
        "S236",
        "S286",
        "S220",
        "S172",
        "S230",
        "S301",
        "S031",
        "S084",
        "S118",
    ]

    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    INTERVALS = ["5m", "15m"]

    WINDOW_BARS = {
        "近7天": {
            "5m": 7 * 24 * 12,
            "15m": 7 * 24 * 4,
        },
        "近30天": {
            "5m": 30 * 24 * 12,
            "15m": 30 * 24 * 4,
        },
    }

    # 这次按最大窗口一次抓够
    TARGET_BARS_BY_INTERVAL = {
        "5m": 30 * 24 * 12 + 800,
        "15m": 30 * 24 * 4 + 400,
    }

    FEE_RATE = 0.0006
    SLIPPAGE = 0.0002
    TOP_N_SCENE = 15
    TOP_N_GLOBAL = 20

    所有场景结果列表 = []
    场景摘要列表 = []
    场景冠军列表 = []
    候选目录表缓存 = []
    全部交易明细列表 = []

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            target_bars = TARGET_BARS_BY_INTERVAL[interval]
            df_raw = 抓取指定bar数量(
                symbol=symbol,
                interval=interval,
                target_bars=target_bars,
            )

            if df_raw.empty:
                print(f"{symbol} {interval} 无数据，跳过。")
                continue

            for window_name, interval_map in WINDOW_BARS.items():
                bars = interval_map[interval]
                df_scene = 切取最近bars(df_raw, bars)

                if len(df_scene) < 300:
                    print(f"{symbol} {interval} {window_name} 数据太少，跳过。")
                    continue

                scene_name = f"{symbol}_{interval}_{window_name}"
                print(f"\n开始场景: {scene_name} | bars={len(df_scene)}")

                report, benchmark, 候选目录表, trade_details_df = 跑候选池场景(
                    df_scene=df_scene,
                    interval=interval,
                    候选策略ID列表=候选策略ID列表,
                    fee_rate=FEE_RATE,
                    slippage=SLIPPAGE,
                )

                report["场景名"] = scene_name
                report["交易对"] = symbol
                report["周期"] = interval
                report["时间窗口"] = window_name
                report["基准收益(%)"] = benchmark
                report["场景内排名"] = np.arange(1, len(report) + 1)

                if not 候选目录表缓存:
                    候选目录表缓存.append(候选目录表)

                if not trade_details_df.empty:
                    trade_details_df["场景名"] = scene_name
                    trade_details_df["交易对"] = symbol
                    trade_details_df["周期"] = interval
                    trade_details_df["时间窗口"] = window_name
                    全部交易明细列表.append(trade_details_df)

                所有场景结果列表.append(report)

                best = report.iloc[0]
                场景冠军列表.append(
                    {
                        "场景名": scene_name,
                        "交易对": symbol,
                        "周期": interval,
                        "时间窗口": window_name,
                        "基准收益(%)": benchmark,
                        "冠军策略ID": best["策略ID"],
                        "冠军策略名称": best["策略名称"],
                        "冠军策略分类": best["策略分类"],
                        "冠军执行版本": best["执行版本"],
                        "冠军收益率(%)": best["收益率(%)"],
                        "冠军最大回撤(%)": best["最大回撤(%)"],
                        "冠军夏普值": best["夏普值"],
                        "冠军综合评分": best["综合评分"],
                        "是否通过基础门槛": best["是否通过基础门槛"],
                    }
                )

                场景摘要列表.append(
                    {
                        "场景名": scene_name,
                        "交易对": symbol,
                        "周期": interval,
                        "时间窗口": window_name,
                        "K线数量": len(df_scene),
                        "基准收益(%)": benchmark,
                        "冠军策略ID": best["策略ID"],
                        "冠军策略名称": best["策略名称"],
                        "冠军执行版本": best["执行版本"],
                        "冠军收益率(%)": best["收益率(%)"],
                        "冠军最大回撤(%)": best["最大回撤(%)"],
                        "冠军夏普值": best["夏普值"],
                        "冠军综合评分": best["综合评分"],
                        "是否通过基础门槛": best["是否通过基础门槛"],
                    }
                )

                print(f"场景完成: {scene_name}")
                print(f"冠军: {best['策略ID']} | {best['执行版本']} | 收益 {best['收益率(%)']:.2f}% | 评分 {best['综合评分']:.2f}")
                print(纯ASCII表格(构建控制台报表(report, top_n=TOP_N_SCENE)))
                print("-" * 120)

    if not 所有场景结果列表:
        print("没有任何场景成功跑完。")
    else:
        所有场景结果 = pd.concat(所有场景结果列表, ignore_index=True)
        场景摘要表 = pd.DataFrame(场景摘要列表)
        场景冠军表 = pd.DataFrame(场景冠军列表)
        候选目录表 = pd.concat(候选目录表缓存, ignore_index=True).drop_duplicates().reset_index(drop=True)

        if 全部交易明细列表:
            全部交易明细 = pd.concat(全部交易明细列表, ignore_index=True)
        else:
            全部交易明细 = pd.DataFrame()

        版本稳定度表 = 汇总版本稳定度(所有场景结果)
        全局控制台报表 = 构建全局控制台报表(版本稳定度表, top_n=TOP_N_GLOBAL)

        print("\n全局版本稳定度 Top 20")
        print(纯ASCII表格(全局控制台报表))

        best_global = 版本稳定度表.iloc[0]
        print("\n全局最稳候选版本")
        print(f"- 策略ID: {best_global['策略ID']}")
        print(f"- 策略名称: {best_global['策略名称']}")
        print(f"- 执行版本: {best_global['执行版本']}")
        print(f"- 策略分类: {best_global['策略分类']}")
        print(f"- 场景数: {best_global['场景数']}")
        print(f"- 平均排名: {best_global['平均排名']:.2f}")
        print(f"- 冠军次数: {best_global['冠军次数']}")
        print(f"- 前三次数: {best_global['前三次数']}")
        print(f"- 前十次数: {best_global['前十次数']}")
        print(f"- 平均收益率: {best_global['平均收益率']:.2f}%")
        print(f"- 平均最大回撤: {best_global['平均最大回撤']:.2f}%")
        print(f"- 平均持仓占比: {best_global['平均持仓占比']:.2f}%")
        print(f"- 平均开仓次数: {best_global['平均开仓次数']:.2f}")
        print(f"- 通过率: {best_global['通过率(%)']:.2f}%")

        paths = 保存输出(
            场景摘要表=场景摘要表,
            所有场景结果=所有场景结果,
            版本稳定度表=版本稳定度表,
            候选目录表=候选目录表,
            全部交易明细=全部交易明细,
            场景冠军表=场景冠军表,
            全局控制台报表=全局控制台报表,
        )

        print("\n已保存文件")
        for k, v in paths.items():
            print(f"- {k}: {v}")
