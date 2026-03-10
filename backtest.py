import os
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import requests

VERSION = "V6.2_Strategy_Lab"
BASE_URL = "https://api.bitget.com/api/v2/mix/market/candles"


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ascii_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"

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


def fetch_safe_5m_data(symbol="BTCUSDT", interval="5m", pages=15, sleep_sec=0.15):
    print(f"[{VERSION}] fetching {symbol} {interval} data ...")

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
                print("no more data, stop fetching.")
                break

            if end_time in seen_end_times:
                print("duplicate endTime detected, stop fetching.")
                break
            seen_end_times.add(end_time)

            new_end_time = str(data[-1][0])
            all_data.extend(data)

            if new_end_time == end_time:
                print("endTime not changed, stop fetching.")
                break

            end_time = new_end_time
            time.sleep(sleep_sec)

        except requests.RequestException as e:
            print(f"request failed: {e}")
            break
        except (ValueError, KeyError, TypeError) as e:
            print(f"parse failed: {e}")
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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def signal_from_conditions(long_cond, short_cond):
    return np.where(long_cond, 1, np.where(short_cond, -1, 0))


def generate_strategy_library(df: pd.DataFrame):
    strategies = {}
    catalog = []

    sid = 1

    def add_strategy(name_cn: str, family: str, params: dict, signal):
        nonlocal sid
        strategy_id = f"S{sid:03d}"
        sid += 1
        strategies[strategy_id] = {
            "name_cn": name_cn,
            "family": family,
            "params": params,
            "signal": signal,
        }
        catalog.append(
            {
                "strategy_id": strategy_id,
                "name_cn": name_cn,
                "family": family,
                "params": str(params),
            }
        )

    # Family A: Trend + RSI + MACD + Volume
    trend_filters = {
        "ema20_gt_50": df["ema_20"] > df["ema_50"],
        "ema20_gt_200": df["ema_20"] > df["ema_200"],
        "close_gt_200": df["close"] > df["ema_200"],
    }
    trend_filters_short = {
        "ema20_gt_50": df["ema_20"] < df["ema_50"],
        "ema20_gt_200": df["ema_20"] < df["ema_200"],
        "close_gt_200": df["close"] < df["ema_200"],
    }

    rsi_pairs = [(55, 45), (60, 40), (65, 35)]
    macd_modes = {
        "macd_cross": (df["macd"] > df["macd_signal"], df["macd"] < df["macd_signal"]),
        "macd_sign": (df["macd"] > 0, df["macd"] < 0),
        "hist_sign": (df["macd_hist"] > 0, df["macd_hist"] < 0),
    }
    volume_modes = {
        "no_vol_filter": (pd.Series(True, index=df.index), pd.Series(True, index=df.index)),
        "vol_gt_ma20": (df["base_vol"] > df["vol_ma_20"], df["base_vol"] > df["vol_ma_20"]),
        "vol_gt_1.5ma20": (df["base_vol"] > df["vol_ma_20"] * 1.5, df["base_vol"] > df["vol_ma_20"] * 1.5),
    }

    for trend_key, (rsi_hi, rsi_lo), macd_key, vol_key in product(
        trend_filters.keys(), rsi_pairs, macd_modes.keys(), volume_modes.keys()
    ):
        long_cond = (
            trend_filters[trend_key]
            & (df["rsi_14"] > rsi_hi)
            & macd_modes[macd_key][0]
            & volume_modes[vol_key][0]
        )
        short_cond = (
            trend_filters_short[trend_key]
            & (df["rsi_14"] < rsi_lo)
            & macd_modes[macd_key][1]
            & volume_modes[vol_key][1]
        )
        add_strategy(
            name_cn=f"趋势确认 {trend_key} RSI{rsi_hi}/{rsi_lo} {macd_key} {vol_key}",
            family="trend_confirm",
            params={
                "trend": trend_key,
                "rsi_hi": rsi_hi,
                "rsi_lo": rsi_lo,
                "macd": macd_key,
                "volume": vol_key,
            },
            signal=signal_from_conditions(long_cond, short_cond),
        )

    # Family B: ATR breakout
    for atr_mult, trend_key, use_vol in product([0.5, 1.0, 1.5], ["ema20_gt_50", "ema20_gt_200"], [False, True]):
        vol_cond = df["base_vol"] > df["vol_ma_20"] if use_vol else pd.Series(True, index=df.index)
        long_cond = (
            (df["close"] > (df["ema_20"] + atr_mult * df["atr_14"]))
            & trend_filters[trend_key]
            & vol_cond
        )
        short_cond = (
            (df["close"] < (df["ema_20"] - atr_mult * df["atr_14"]))
            & trend_filters_short[trend_key]
            & vol_cond
        )
        add_strategy(
            name_cn=f"ATR突破 {trend_key} x{atr_mult} vol={use_vol}",
            family="atr_breakout",
            params={"trend": trend_key, "atr_mult": atr_mult, "use_vol": use_vol},
            signal=signal_from_conditions(long_cond, short_cond),
        )

    # Family C: Mean reversion RSI
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

        add_strategy(
            name_cn=f"RSI均值回归 {low}/{high} trend={with_trend}",
            family="rsi_mean_revert",
            params={"low": low, "high": high, "with_trend": with_trend},
            signal=signal_from_conditions(long_cond, short_cond),
        )

    # Family D: Bollinger mean reversion
    for with_trend in [False, True]:
        long_cond = df["close"] < df["bb_lower_2"]
        short_cond = df["close"] > df["bb_upper_2"]
        if with_trend:
            long_cond = long_cond & (df["close"] > df["ema_200"])
            short_cond = short_cond & (df["close"] < df["ema_200"])

        add_strategy(
            name_cn=f"布林回归 trend={with_trend}",
            family="bollinger_revert",
            params={"with_trend": with_trend},
            signal=signal_from_conditions(long_cond, short_cond),
        )

    # Family E: EMA distance grid
    for pct, base in product([0.005, 0.01, 0.015, 0.02], ["ema_20", "ema_50"]):
        long_cond = df["close"] < df[base] * (1 - pct)
        short_cond = df["close"] > df[base] * (1 + pct)
        add_strategy(
            name_cn=f"均线偏离 {base} {pct:.3f}",
            family="ema_distance",
            params={"base": base, "pct": pct},
            signal=signal_from_conditions(long_cond, short_cond),
        )

    # Family F: Hybrid filters
    hybrid_configs = [
        ("close_gt_200", "macd_cross", "vol_gt_ma20", 55, 45),
        ("ema20_gt_50", "hist_sign", "vol_gt_ma20", 60, 40),
        ("ema20_gt_200", "macd_sign", "vol_gt_1.5ma20", 55, 45),
        ("close_gt_200", "hist_sign", "no_vol_filter", 65, 35),
    ]
    for trend_key, macd_key, vol_key, rsi_hi, rsi_lo in hybrid_configs:
        long_cond = (
            trend_filters[trend_key]
            & macd_modes[macd_key][0]
            & volume_modes[vol_key][0]
            & (df["rsi_14"] > rsi_hi)
            & (df["close"] > df["ema_20"])
        )
        short_cond = (
            trend_filters_short[trend_key]
            & macd_modes[macd_key][1]
            & volume_modes[vol_key][1]
            & (df["rsi_14"] < rsi_lo)
            & (df["close"] < df["ema_20"])
        )
        add_strategy(
            name_cn=f"混合共振 {trend_key} {macd_key} {vol_key} {rsi_hi}/{rsi_lo}",
            family="hybrid",
            params={
                "trend": trend_key,
                "macd": macd_key,
                "volume": vol_key,
                "rsi_hi": rsi_hi,
                "rsi_lo": rsi_lo,
            },
            signal=signal_from_conditions(long_cond, short_cond),
        )

    return strategies, pd.DataFrame(catalog)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return float(drawdown.min())


def calculate_sharpe(net_ret: pd.Series, bars_per_year=12 * 24 * 365) -> float:
    std = net_ret.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((net_ret.mean() / std) * np.sqrt(bars_per_year))


def calculate_calmar(total_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return float(total_return / abs(max_dd))


def backtest_one_strategy(df: pd.DataFrame, signal, fee_rate=0.0006, slippage=0.0002):
    raw_signal = pd.Series(signal, index=df.index)

    pos = raw_signal.replace(0, np.nan).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)

    turnover = pos.diff().abs().fillna(pos.abs())
    trading_cost = turnover * (fee_rate + slippage)

    gross_ret = pos * df["ret"]
    net_ret = gross_ret - trading_cost
    equity_curve = (1 + net_ret).cumprod()

    total_return = float(equity_curve.iloc[-1] - 1)
    max_dd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe(net_ret)
    calmar = calculate_calmar(total_return, max_dd)

    trade_entries = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())
    direction_changes = int(((pos * pos.shift(1).fillna(0)) < 0).sum())
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
            "return_pct": total_return * 100,
            "maxdd_pct": max_dd * 100,
            "sharpe": sharpe,
            "calmar": calmar,
            "entries": trade_entries,
            "flips": direction_changes,
            "exposure_pct": exposure * 100,
            "winrate_pct": win_rate * 100,
            "pnl_ratio": pnl_ratio,
        },
    }


def add_score(report: pd.DataFrame) -> pd.DataFrame:
    df = report.copy()

    df["score"] = (
        df["return_pct"] * 0.45
        + df["sharpe"] * 2.0
        + df["calmar"] * 8.0
        + df["winrate_pct"] * 0.03
        + df["pnl_ratio"] * 4.0
        - df["maxdd_pct"].abs() * 0.35
    )
    return df


def run_strategy_lab(df, fee_rate=0.0006, slippage=0.0002):
    df = calculate_indicators(df)
    strategies, catalog_df = generate_strategy_library(df)

    results = []
    equity_table = pd.DataFrame({"timestamp": df["timestamp"]})

    for strategy_id, meta in strategies.items():
        result = backtest_one_strategy(df, meta["signal"], fee_rate=fee_rate, slippage=slippage)
        summary = result["summary"]
        summary["strategy_id"] = strategy_id
        summary["family"] = meta["family"]
        summary["name_cn"] = meta["name_cn"]
        results.append(summary)
        equity_table[strategy_id] = result["equity_curve"].values

    report = pd.DataFrame(results)
    report = add_score(report)
    report = report.sort_values(by="score", ascending=False).reset_index(drop=True)

    benchmark = ((1 + df["ret"]).cumprod().iloc[-1] - 1) * 100
    return report, benchmark, equity_table, catalog_df


def build_console_report(report: pd.DataFrame, top_n=20):
    view = report.head(top_n).copy()

    cols = [
        "strategy_id",
        "family",
        "return_pct",
        "maxdd_pct",
        "sharpe",
        "calmar",
        "entries",
        "flips",
        "exposure_pct",
        "winrate_pct",
        "pnl_ratio",
        "score",
    ]
    view = view[cols]

    rename_map = {
        "strategy_id": "ID",
        "family": "Family",
        "return_pct": "Return%",
        "maxdd_pct": "MaxDD%",
        "sharpe": "Sharpe",
        "calmar": "Calmar",
        "entries": "Entries",
        "flips": "Flips",
        "exposure_pct": "Exposure%",
        "winrate_pct": "WinRate%",
        "pnl_ratio": "PnLRatio",
        "score": "Score",
    }
    view = view.rename(columns=rename_map)

    for col in ["Return%", "MaxDD%", "Exposure%", "WinRate%"]:
        view[col] = view[col].map(lambda x: f"{x:.2f}%")

    for col in ["Sharpe", "Calmar", "PnLRatio", "Score"]:
        view[col] = view[col].map(lambda x: f"{x:.2f}")

    return view


def build_markdown_report(report: pd.DataFrame, benchmark: float, top_n=30) -> str:
    best = report.iloc[0]

    md = []
    md.append(f"# {VERSION} Backtest Report")
    md.append("")
    md.append(f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- Benchmark: {benchmark:.2f}%")
    md.append(f"- Total Strategies: {len(report)}")
    md.append(f"- Best Strategy ID: {best['strategy_id']}")
    md.append(f"- Best Strategy Name: {best['name_cn']}")
    md.append(f"- Best Score: {best['score']:.2f}")
    md.append("")

    top_view = report.head(top_n).copy()
    top_view = top_view[
        [
            "strategy_id",
            "family",
            "name_cn",
            "return_pct",
            "maxdd_pct",
            "sharpe",
            "calmar",
            "entries",
            "flips",
            "exposure_pct",
            "winrate_pct",
            "pnl_ratio",
            "score",
        ]
    ]

    top_view = top_view.rename(
        columns={
            "strategy_id": "ID",
            "family": "Family",
            "name_cn": "StrategyName",
            "return_pct": "Return%",
            "maxdd_pct": "MaxDD%",
            "sharpe": "Sharpe",
            "calmar": "Calmar",
            "entries": "Entries",
            "flips": "Flips",
            "exposure_pct": "Exposure%",
            "winrate_pct": "WinRate%",
            "pnl_ratio": "PnLRatio",
            "score": "Score",
        }
    )

    try:
        md.append(top_view.to_markdown(index=False, floatfmt=".2f"))
    except Exception:
        md.append(top_view.to_string(index=False))

    md.append("")
    return "\n".join(md)


def save_outputs(report, benchmark, equity_table, catalog_df, console_report, symbol, interval):
    ts = now_str()
    out_dir = os.path.join("outputs", ts)
    ensure_dir(out_dir)

    report_path = os.path.join(out_dir, f"report_{symbol}_{interval}_{ts}.csv")
    equity_path = os.path.join(out_dir, f"equity_curves_{symbol}_{interval}_{ts}.csv")
    catalog_path = os.path.join(out_dir, f"strategy_catalog_{symbol}_{interval}_{ts}.csv")
    summary_txt_path = os.path.join(out_dir, f"summary_{symbol}_{interval}_{ts}.txt")
    summary_md_path = os.path.join(out_dir, f"summary_{symbol}_{interval}_{ts}.md")

    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    equity_table.to_csv(equity_path, index=False, encoding="utf-8-sig")
    catalog_df.to_csv(catalog_path, index=False, encoding="utf-8-sig")

    with open(summary_txt_path, "w", encoding="utf-8-sig") as f:
        f.write(f"[{VERSION}] Strategy Lab\n")
        f.write(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Interval: {interval}\n")
        f.write(f"Benchmark: {benchmark:.2f}%\n")
        f.write(f"Total Strategies: {len(report)}\n\n")
        f.write(ascii_table(console_report))
        f.write("\n")

    with open(summary_md_path, "w", encoding="utf-8-sig") as f:
        f.write(build_markdown_report(report, benchmark, top_n=30))

    return {
        "out_dir": out_dir,
        "report_path": report_path,
        "equity_path": equity_path,
        "catalog_path": catalog_path,
        "summary_txt_path": summary_txt_path,
        "summary_md_path": summary_md_path,
    }


if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    INTERVAL = "5m"
    PAGES = 15
    FEE_RATE = 0.0006
    SLIPPAGE = 0.0002
    TOP_N_CONSOLE = 20

    df = fetch_safe_5m_data(symbol=SYMBOL, interval=INTERVAL, pages=PAGES)

    if df.empty:
        print("no data fetched.")
    else:
        report, benchmark, equity_table, catalog_df = run_strategy_lab(
            df=df,
            fee_rate=FEE_RATE,
            slippage=SLIPPAGE,
        )

        console_report = build_console_report(report, top_n=TOP_N_CONSOLE)

        print(f"\n[{VERSION}] Strategy Lab Result")
        print(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Benchmark: {benchmark:.2f}%")
        print(f"Total Strategies: {len(report)}")
        print()
        print(ascii_table(console_report))

        paths = save_outputs(
            report=report,
            benchmark=benchmark,
            equity_table=equity_table,
            catalog_df=catalog_df,
            console_report=console_report,
            symbol=SYMBOL,
            interval=INTERVAL,
        )

        best = report.iloc[0]
        print("\nBest Strategy")
        print(f"- ID: {best['strategy_id']}")
        print(f"- Name: {best['name_cn']}")
        print(f"- Score: {best['score']:.2f}")
        print(f"- Return: {best['return_pct']:.2f}%")
        print(f"- MaxDD: {best['maxdd_pct']:.2f}%")
        print(f"- Sharpe: {best['sharpe']:.2f}")

        print("\nSaved Files")
        for k, v in paths.items():
            print(f"- {k}: {v}")
