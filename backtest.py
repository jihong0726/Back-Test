import os
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import requests

VERSION = "V6.3_中文策略实验室"
BASE_URL = "https://api.bitget.com/api/v2/mix/market/candles"


def 当前时间字符串():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def 创建目录(path: str):
    os.makedirs(path, exist_ok=True)


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
                print("没有更多数据，停止抓取。")
                break

            if end_time in seen_end_times:
                print("endTime 重复，停止抓取。")
                break
            seen_end_times.add(end_time)

            new_end_time = str(data[-1][0])
            all_data.extend(data)

            if new_end_time == end_time:
                print("endTime 未变化，停止抓取。")
                break

            end_time = new_end_time
            time.sleep(sleep_sec)

        except requests.RequestException as e:
            print(f"请求失败: {e}")
            break
        except (ValueError, KeyError, TypeError) as e:
            print(f"数据解析失败: {e}")
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

    for low, high, with_trend in product([20, 25, 30], [70, 75, 80], [False, True]):
        if low >= high:
            continue
        base_long = df["rsi_14"] < low
        base_short = df["rsi_14"] > high

        if with_trend:
            long_cond = base_long & (df["close"] > df["ema_200"])
            short_cond = base_short & (df["close"] < df["ema_200"])
        else:
            long_cond = base_long
            short_cond = base_short

        新增策略(
            name_cn=f"RSI均值回归 {low}/{high} 趋势过滤={with_trend}",
            family_cn="RSI均值回归类",
            params={"RSI低点": low, "RSI高点": high, "趋势过滤": with_trend},
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

    for pct, base in product([0.005, 0.01, 0.015, 0.02], ["ema_20", "ema_50"]):
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


def 夏普(net_ret: pd.Series, bars_per_year=12 * 24 * 365) -> float:
    std = net_ret.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((net_ret.mean() / std) * np.sqrt(bars_per_year))


def 卡玛(total_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return float(total_return / abs(max_dd))


def 回测单策略(df: pd.DataFrame, signal, fee_rate=0.0006, slippage=0.0002):
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
    sharpe = 夏普(net_ret)
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
        df["收益率(%)"] * 0.45
        + df["夏普值"] * 2.0
        + df["卡玛值"] * 8.0
        + df["胜率(%)"] * 0.03
        + df["盈亏比"] * 4.0
        - df["最大回撤(%)"].abs() * 0.35
    )
    return df


def 跑策略实验室(df, fee_rate=0.0006, slippage=0.0002):
    df = 计算指标(df)
    strategies, catalog_df = 生成策略库(df)

    results = []
    equity_table = pd.DataFrame({"时间": df["timestamp"]})

    for strategy_id, meta in strategies.items():
        result = 回测单策略(df, meta["signal"], fee_rate=fee_rate, slippage=slippage)
        summary = result["summary"]
        summary["策略ID"] = strategy_id
        summary["策略分类"] = meta["family_cn"]
        summary["策略名称"] = meta["name_cn"]
        results.append(summary)
        equity_table[strategy_id] = result["equity_curve"].values

    report = pd.DataFrame(results)
    report = 计算综合评分(report)
    report = report.sort_values(by="综合评分", ascending=False).reset_index(drop=True)

    benchmark = ((1 + df["ret"]).cumprod().iloc[-1] - 1) * 100
    return report, benchmark, equity_table, catalog_df


def 构建控制台报表(report: pd.DataFrame, top_n=20):
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
        ]
    ]

    for col in ["收益率(%)", "最大回撤(%)", "持仓占比(%)", "胜率(%)"]:
        view[col] = view[col].map(lambda x: f"{x:.2f}%")

    for col in ["夏普值", "卡玛值", "盈亏比", "综合评分"]:
        view[col] = view[col].map(lambda x: f"{x:.2f}")

    return view


def 构建Markdown报告(report: pd.DataFrame, benchmark: float, top_n=30) -> str:
    best = report.iloc[0]

    md = []
    md.append(f"# {VERSION} 回测报告")
    md.append("")
    md.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- 基准收益: {benchmark:.2f}%")
    md.append(f"- 策略总数: {len(report)}")
    md.append(f"- 最佳策略ID: {best['策略ID']}")
    md.append(f"- 最佳策略名称: {best['策略名称']}")
    md.append(f"- 最佳综合评分: {best['综合评分']:.2f}")
    md.append("")

    top_view = report.head(top_n).copy()
    try:
        md.append(top_view.to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        md.append(top_view.to_string(index=False))

    md.append("")
    return "\n".join(md)


def 保存输出(report, benchmark, equity_table, catalog_df, console_report, symbol, interval):
    ts = 当前时间字符串()
    out_dir = os.path.join("outputs", ts)
    创建目录(out_dir)

    report_path = os.path.join(out_dir, f"回测结果_{symbol}_{interval}_{ts}.csv")
    equity_path = os.path.join(out_dir, f"资金曲线_{symbol}_{interval}_{ts}.csv")
    catalog_path = os.path.join(out_dir, f"策略目录_{symbol}_{interval}_{ts}.csv")
    summary_txt_path = os.path.join(out_dir, f"回测摘要_{symbol}_{interval}_{ts}.txt")
    summary_md_path = os.path.join(out_dir, f"回测摘要_{symbol}_{interval}_{ts}.md")

    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    equity_table.to_csv(equity_path, index=False, encoding="utf-8-sig")
    catalog_df.to_csv(catalog_path, index=False, encoding="utf-8-sig")

    with open(summary_txt_path, "w", encoding="utf-8-sig") as f:
        f.write(f"[{VERSION}] 中文策略实验室\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"交易对: {symbol}\n")
        f.write(f"周期: {interval}\n")
        f.write(f"基准收益: {benchmark:.2f}%\n")
        f.write(f"策略总数: {len(report)}\n\n")
        f.write(纯ASCII表格(console_report))
        f.write("\n")

    with open(summary_md_path, "w", encoding="utf-8-sig") as f:
        f.write(构建Markdown报告(report, benchmark, top_n=30))

    return {
        "输出目录": out_dir,
        "回测结果文件": report_path,
        "资金曲线文件": equity_path,
        "策略目录文件": catalog_path,
        "摘要TXT文件": summary_txt_path,
        "摘要MD文件": summary_md_path,
    }


if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    INTERVAL = "5m"
    PAGES = 15
    FEE_RATE = 0.0006
    SLIPPAGE = 0.0002
    TOP_N_CONSOLE = 20

    df = 抓取K线数据(symbol=SYMBOL, interval=INTERVAL, pages=PAGES)

    if df.empty:
        print("没有成功获取到数据。")
    else:
        report, benchmark, equity_table, catalog_df = 跑策略实验室(
            df=df,
            fee_rate=FEE_RATE,
            slippage=SLIPPAGE,
        )

        console_report = 构建控制台报表(report, top_n=TOP_N_CONSOLE)

        print(f"\n[{VERSION}] 策略实验室结果")
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"基准收益: {benchmark:.2f}%")
        print(f"策略总数: {len(report)}")
        print()
        print(纯ASCII表格(console_report))

        paths = 保存输出(
            report=report,
            benchmark=benchmark,
            equity_table=equity_table,
            catalog_df=catalog_df,
            console_report=console_report,
            symbol=SYMBOL,
            interval=INTERVAL,
        )

        best = report.iloc[0]
        print("\n最佳策略")
        print(f"- 策略ID: {best['策略ID']}")
        print(f"- 策略名称: {best['策略名称']}")
        print(f"- 综合评分: {best['综合评分']:.2f}")
        print(f"- 收益率: {best['收益率(%)']:.2f}%")
        print(f"- 最大回撤: {best['最大回撤(%)']:.2f}%")
        print(f"- 夏普值: {best['夏普值']:.2f}")

        print("\n已保存文件")
        for k, v in paths.items():
            print(f"- {k}: {v}")
