import os
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import requests

VERSION = "V6.4_多市场策略实验室"
BASE_URL = "https://api.bitget.com/api/v2/mix/market/candles"


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


def 抓取K线数据(symbol="BTCUSDT", interval="5m", pages=15, sleep_sec=0.15):
    print(f"[{VERSION}] 正在抓取 {symbol} {interval} 数据...")

    end_time = str(int(time.time() * 1000))
    all_data = []
    seen_end_times = set()

    for _ in range(pages):
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

            new_end_time = str(data[-1][0])
            all_data.extend(data)

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

    numeric_cols = ["open", "high", "low", "close", "base_vol", "quote_vol"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def 切取最近bars(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if len(df) <= bars:
        return df.copy()
    return df.tail(bars).reset_index(drop=True)


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
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    for rsi_period in [7, 10, 14, 21]:
        delta_p = df["close"].diff()
        gain_p = delta_p.clip(lower=0)
        loss_p = -delta_p.clip(upper=0)
        avg_gain_p = gain_p.ewm(alpha=1 / rsi_period, adjust=False).mean()
        avg_loss_p = loss_p.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs_p = avg_gain_p / avg_loss_p.replace(0, np.nan)
        df[f"rsi_{rsi_period}"] = 100 - (100 / (1 + rs_p))

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

    for trend_key, (rsi_hi, rsi_lo), macd_key, vol_key in product(
        趋势多.keys(), rsi_pairs, macd_modes.keys(), volume_modes.keys()
    ):
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

    for atr_mult, trend_key, use_vol in product([0.5, 1.0, 1.5], ["ema20上穿ema50", "ema20上穿ema200"], [False, True]):
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

    for rsi_period, low, high, with_trend in product([7, 10, 14, 21], [18, 20, 22, 25, 30], [70, 75, 78, 80, 82], [False, True]):
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
            params={"RSI周期": rsi_period, "RSI低点": low, "RSI高点": high, "趋势过滤": with_trend},
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

    for pct, base in product([0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020], ["ema_20", "ema_50"]):
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


def 周期年化bar数(interval: str) -> int:
    mapping = {
        "3m": 20 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "30m": 2 * 24 * 365,
        "1H": 24 * 365,
        "4H": 6 * 365,
    }
    return mapping.get(interval, 12 * 24 * 365)


def 回测单策略(df: pd.DataFrame, signal, fee_rate=0.0006, slippage=0.0002, interval="5m"):
    raw_signal = pd.Series(signal, index=df.index)
    pos = raw_signal.replace(0, np.nan).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)

    turnover = pos.diff().abs().fillna(pos.abs())
    trading_cost = turnover * (fee_rate + slippage)

    gross_ret = pos * df["ret"]
    net_ret = gross_ret - trading_cost
    equity_curve = (1 + net_ret).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1)
    max_dd = 最大回撤(equity_curve)
    sharpe = 夏普(net_ret, 周期年化bar数(interval))
    calmar = 卡玛(total_return, max_dd)

    entries = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())
    flips = int(((pos * pos.shift(1).fillna(0)) < 0).sum())
    exposure = float((pos != 0).mean())

    non_zero_ret = net_ret[net_ret != 0]
    win_rate = float((non_zero_ret > 0).mean()) if len(non_zero_ret) > 0 else 0.0

    avg_win = net_ret[net_ret > 0].mean()
    avg_loss = net_ret[net_ret < 0].mean()
    pnl_ratio = (
        float(abs(avg_win / avg_loss))
        if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
        else 0.0
    )

    return {
        "position": pos,
        "net_ret": net_ret,
        "equity_curve": equity_curve,
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
        (df["开仓次数"] >= 3)
        & (df["持仓占比(%)"] <= 95)
        & (df["最大回撤(%)"] >= -20)
    )

    return df


def 跑单场景(df: pd.DataFrame, interval: str, fee_rate=0.0006, slippage=0.0002):
    df = 计算指标(df)
    strategies, catalog_df = 生成策略库(df)

    results = []
    equity_table = pd.DataFrame({"时间": df["timestamp"]})

    for strategy_id, meta in strategies.items():
        result = 回测单策略(
            df=df,
            signal=meta["signal"],
            fee_rate=fee_rate,
            slippage=slippage,
            interval=interval,
        )
        summary = result["summary"]
        summary["策略ID"] = strategy_id
        summary["策略分类"] = meta["family_cn"]
        summary["策略名称"] = meta["name_cn"]
        results.append(summary)
        equity_table[strategy_id] = result["equity_curve"].values

    report = pd.DataFrame(results)
    report = 计算综合评分(report)

    report = report.sort_values(
        by=["是否通过基础门槛", "综合评分"],
        ascending=[False, False]
    ).reset_index(drop=True)

    benchmark = ((1 + df["ret"]).cumprod().iloc[-1] - 1) * 100
    return report, benchmark, equity_table, catalog_df


def 构建控制台报表(report: pd.DataFrame, top_n=15):
    view = report.head(top_n).copy()
    view = view[
        [
            "策略ID",
            "策略分类",
            "收益率(%)",
            "最大回撤(%)",
            "夏普值",
            "卡玛值",
            "开仓次数",
            "反手次数",
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


def 汇总稳定度(所有场景结果: pd.DataFrame) -> pd.DataFrame:
    df = 所有场景结果.copy()

    df["进前10"] = df["场景内排名"] <= 10
    df["进前3"] = df["场景内排名"] <= 3
    df["冠军"] = df["场景内排名"] == 1

    grouped = df.groupby(["策略ID", "策略名称", "策略分类"], as_index=False).agg(
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
        平均评分=("综合评分", "mean"),
        通过门槛次数=("是否通过基础门槛", "sum"),
    )

    grouped["通过率(%)"] = grouped["通过门槛次数"] / grouped["场景数"] * 100
    grouped["前十率(%)"] = grouped["前十次数"] / grouped["场景数"] * 100
    grouped["前三率(%)"] = grouped["前三次数"] / grouped["场景数"] * 100
    grouped["冠军率(%)"] = grouped["冠军次数"] / grouped["场景数"] * 100

    grouped = grouped.sort_values(
        by=["冠军次数", "前三次数", "前十次数", "平均评分"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return grouped


def 构建全局控制台报表(稳定度表: pd.DataFrame, top_n=20):
    view = 稳定度表.head(top_n).copy()
    view = view[
        [
            "策略ID",
            "策略分类",
            "场景数",
            "平均排名",
            "前十次数",
            "前三次数",
            "冠军次数",
            "平均收益率",
            "平均最大回撤",
            "平均夏普",
            "平均评分",
            "通过率(%)",
        ]
    ]

    for col in ["平均排名", "平均收益率", "平均最大回撤", "平均夏普", "平均评分", "通过率(%)"]:
        if col == "通过率(%)":
            view[col] = view[col].map(百分比格式)
        else:
            view[col] = view[col].map(数字格式)

    return view


def 保存输出(
    场景摘要表,
    所有场景结果,
    稳定度表,
    策略目录表,
    全局控制台报表,
    场景冠军表,
):
    ts = 当前时间字符串()
    out_dir = os.path.join("outputs", ts)
    创建目录(out_dir)

    路径 = {
        "输出目录": out_dir,
        "场景摘要文件": os.path.join(out_dir, f"场景摘要_{ts}.csv"),
        "全部结果文件": os.path.join(out_dir, f"全部场景策略结果_{ts}.csv"),
        "稳定度统计文件": os.path.join(out_dir, f"策略稳定度统计_{ts}.csv"),
        "策略目录文件": os.path.join(out_dir, f"策略目录_{ts}.csv"),
        "场景冠军文件": os.path.join(out_dir, f"场景冠军列表_{ts}.csv"),
        "摘要TXT文件": os.path.join(out_dir, f"总摘要_{ts}.txt"),
        "摘要MD文件": os.path.join(out_dir, f"总摘要_{ts}.md"),
    }

    场景摘要表.to_csv(路径["场景摘要文件"], index=False, encoding="utf-8-sig")
    所有场景结果.to_csv(路径["全部结果文件"], index=False, encoding="utf-8-sig")
    稳定度表.to_csv(路径["稳定度统计文件"], index=False, encoding="utf-8-sig")
    策略目录表.to_csv(路径["策略目录文件"], index=False, encoding="utf-8-sig")
    场景冠军表.to_csv(路径["场景冠军文件"], index=False, encoding="utf-8-sig")

    with open(路径["摘要TXT文件"], "w", encoding="utf-8-sig") as f:
        f.write(f"[{VERSION}] 多市场多周期回测总摘要\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总场景数: {len(场景摘要表)}\n")
        f.write(f"策略总数: {稳定度表['策略ID'].nunique()}\n\n")
        f.write("全局稳定度 Top 20\n")
        f.write(纯ASCII表格(全局控制台报表))
        f.write("\n\n场景冠军列表\n")
        f.write(纯ASCII表格(场景冠军表.head(20)))
        f.write("\n")

    with open(路径["摘要MD文件"], "w", encoding="utf-8-sig") as f:
        f.write(f"# {VERSION} 多市场多周期回测总摘要\n\n")
        f.write(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 总场景数: {len(场景摘要表)}\n")
        f.write(f"- 策略总数: {稳定度表['策略ID'].nunique()}\n\n")
        f.write("## 全局稳定度 Top 20\n\n")
        try:
            f.write(全局控制台报表.to_markdown(index=False))
        except Exception:
            f.write(全局控制台报表.to_string(index=False))
        f.write("\n\n## 场景冠军列表\n\n")
        try:
            f.write(场景冠军表.head(20).to_markdown(index=False))
        except Exception:
            f.write(场景冠军表.head(20).to_string(index=False))
        f.write("\n")

    return 路径


if __name__ == "__main__":
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

    FETCH_PAGES = {
        "5m": 12,
        "15m": 12,
    }

    FEE_RATE = 0.0006
    SLIPPAGE = 0.0002
    TOP_N_SCENE = 15
    TOP_N_GLOBAL = 20

    所有场景结果列表 = []
    场景摘要列表 = []
    场景冠军列表 = []
    全部策略目录 = []

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            df_raw = 抓取K线数据(
                symbol=symbol,
                interval=interval,
                pages=FETCH_PAGES.get(interval, 12),
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

                report, benchmark, equity_table, catalog_df = 跑单场景(
                    df=df_scene,
                    interval=interval,
                    fee_rate=FEE_RATE,
                    slippage=SLIPPAGE,
                )

                report["场景名"] = scene_name
                report["交易对"] = symbol
                report["周期"] = interval
                report["时间窗口"] = window_name
                report["基准收益(%)"] = benchmark
                report["场景内排名"] = np.arange(1, len(report) + 1)

                if not 全部策略目录:
                    全部策略目录.append(catalog_df)

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
                        "冠军策略分类": best["策略分类"],
                        "冠军收益率(%)": best["收益率(%)"],
                        "冠军最大回撤(%)": best["最大回撤(%)"],
                        "冠军夏普值": best["夏普值"],
                        "冠军综合评分": best["综合评分"],
                    }
                )

                print(f"场景完成: {scene_name}")
                print(f"冠军: {best['策略ID']} | {best['策略名称']} | 收益 {best['收益率(%)']:.2f}% | 评分 {best['综合评分']:.2f}")
                print(纯ASCII表格(构建控制台报表(report, top_n=TOP_N_SCENE)))
                print("-" * 120)

    if not 所有场景结果列表:
        print("没有任何场景成功跑完。")
    else:
        所有场景结果 = pd.concat(所有场景结果列表, ignore_index=True)
        场景摘要表 = pd.DataFrame(场景摘要列表)
        场景冠军表 = pd.DataFrame(场景冠军列表)
        策略目录表 = pd.concat(全部策略目录, ignore_index=True).drop_duplicates().reset_index(drop=True)

        稳定度表 = 汇总稳定度(所有场景结果)
        全局控制台报表 = 构建全局控制台报表(稳定度表, top_n=TOP_N_GLOBAL)

        print("\n全局稳定度 Top 20")
        print(纯ASCII表格(全局控制台报表))

        best_global = 稳定度表.iloc[0]
        print("\n全局最稳策略")
        print(f"- 策略ID: {best_global['策略ID']}")
        print(f"- 策略名称: {best_global['策略名称']}")
        print(f"- 策略分类: {best_global['策略分类']}")
        print(f"- 场景数: {best_global['场景数']}")
        print(f"- 平均排名: {best_global['平均排名']:.2f}")
        print(f"- 冠军次数: {best_global['冠军次数']}")
        print(f"- 前三次数: {best_global['前三次数']}")
        print(f"- 前十次数: {best_global['前十次数']}")
        print(f"- 平均收益率: {best_global['平均收益率']:.2f}%")
        print(f"- 平均最大回撤: {best_global['平均最大回撤']:.2f}%")
        print(f"- 平均评分: {best_global['平均评分']:.2f}")

        路径 = 保存输出(
            场景摘要表=场景摘要表,
            所有场景结果=所有场景结果,
            稳定度表=稳定度表,
            策略目录表=策略目录表,
            全局控制台报表=全局控制台报表,
            场景冠军表=场景冠军表,
        )

        print("\n已保存文件")
        for k, v in 路径.items():
            print(f"- {k}: {v}")
