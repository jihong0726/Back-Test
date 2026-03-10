import time
import requests
import pandas as pd
import numpy as np

VERSION = "V6.1_Backtest_Reliable"

BASE_URL = "https://api.bitget.com/api/v2/mix/market/candles"


def fetch_safe_5m_data(symbol="BTCUSDT", interval="5m", pages=15, sleep_sec=0.15):
    print(f"[{VERSION}] 抓取 {symbol} {interval} 数据中...")

    end_time = str(int(time.time() * 1000))
    all_data = []

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
                print("没有拿到更多数据，停止抓取。")
                break

            new_end_time = data[-1][0]
            if str(new_end_time) == str(end_time):
                print("endTime 未变化，停止抓取。")
                break

            all_data.extend(data)
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


def calculate_indicators(df):
    df = df.copy()

    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # Volume MA
    df["vol_ma"] = df["base_vol"].rolling(20).mean()

    # ATR
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_s"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Return
    df["m_ret"] = df["close"].pct_change().fillna(0)

    return df


def build_strategies(df):
    strategies = {}

    strategies["1.精英趋势(EMA+RSI+VOL)"] = np.where(
        (df["close"] > df["ema_200"]) & (df["rsi"] > 55) & (df["base_vol"] > df["vol_ma"]),
        1,
        np.where(
            (df["close"] < df["ema_200"]) & (df["rsi"] < 45) & (df["base_vol"] > df["vol_ma"]),
            -1,
            0,
        ),
    )

    strategies["2.波动率突破(EMA+ATR)"] = np.where(
        df["close"] > (df["ema_20"] + df["atr"]),
        1,
        np.where(df["close"] < (df["ema_20"] - df["atr"]), -1, 0),
    )

    strategies["3.极速动能(MACD+RSI60)"] = np.where(
        (df["macd"] > df["macd_s"]) & (df["rsi"] > 60),
        1,
        np.where((df["macd"] < df["macd_s"]) & (df["rsi"] < 40), -1, 0),
    )

    strategies["4.机构回调(EMA200抄底)"] = np.where(
        (df["close"] > df["ema_200"]) & (df["rsi"] < 35),
        1,
        np.where((df["close"] < df["ema_200"]) & (df["rsi"] > 65), -1, 0),
    )

    strategies["5.量能背离(逆势预警)"] = np.where(
        (df["close"] > df["close"].shift(20)) & (df["base_vol"] < df["vol_ma"]),
        -1,
        np.where((df["close"] < df["close"].shift(20)) & (df["base_vol"] < df["vol_ma"]), 1, 0),
    )

    strategies["6.纠缠突破(EMA20/200)"] = np.where(
        (df["ema_20"] > df["ema_200"]) & (df["base_vol"] > df["vol_ma"] * 1.5),
        1,
        np.where(
            (df["ema_20"] < df["ema_200"]) & (df["base_vol"] > df["vol_ma"] * 1.5),
            -1,
            0,
        ),
    )

    strategies["7.暴力RSI(80/20)"] = np.where(
        df["rsi"] < 20,
        1,
        np.where(df["rsi"] > 80, -1, 0),
    )

    strategies["8.ATR守卫(趋势追踪)"] = np.where(
        df["close"] > (df["close"].shift(1) + df["atr"]),
        1,
        np.where(df["close"] < (df["close"].shift(1) - df["atr"]), -1, 0),
    )

    strategies["9.智能网格(围绕EMA20)"] = np.where(
        df["close"] < df["ema_20"] * 0.99,
        1,
        np.where(df["close"] > df["ema_20"] * 1.01, -1, 0),
    )

    strategies["10.全因子共振(四重过滤)"] = np.where(
        (df["rsi"] > 50)
        & (df["macd"] > 0)
        & (df["base_vol"] > df["vol_ma"])
        & (df["close"] > df["ema_200"]),
        1,
        np.where(
            (df["rsi"] < 50)
            & (df["macd"] < 0)
            & (df["base_vol"] > df["vol_ma"])
            & (df["close"] < df["ema_200"]),
            -1,
            0,
        ),
    )

    return strategies


def calculate_max_drawdown(equity_curve):
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return drawdown.min()


def calculate_sharpe(net_ret, bars_per_year=12 * 24 * 365):
    std = net_ret.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return (net_ret.mean() / std) * np.sqrt(bars_per_year)


def backtest_one_strategy(df, signal, fee_rate=0.0006, slippage=0.0002):
    raw_signal = pd.Series(signal, index=df.index)

    # 保留仓位直到反向/平仓信号
    pos = raw_signal.replace(0, np.nan).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)

    # turnover: 0->1 = 1, 1->-1 = 2
    turnover = pos.diff().abs().fillna(pos.abs())

    trading_cost = turnover * (fee_rate + slippage)
    gross_ret = pos * df["m_ret"]
    net_ret = gross_ret - trading_cost
    equity_curve = (1 + net_ret).cumprod()

    total_return = equity_curve.iloc[-1] - 1
    max_dd = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe(net_ret)

    trade_entries = ((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum()
    direction_changes = ((pos * pos.shift(1).fillna(0)) < 0).sum()
    exposure = (pos != 0).mean()

    win_rate = (net_ret[net_ret != 0] > 0).mean()
    if pd.isna(win_rate):
        win_rate = 0.0

    avg_win = net_ret[net_ret > 0].mean()
    avg_loss = net_ret[net_ret < 0].mean()
    pnl_ratio = abs(avg_win / avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else 0.0

    return {
        "position": pos,
        "net_ret": net_ret,
        "equity_curve": equity_curve,
        "summary": {
            "净收益(%)": total_return * 100,
            "最大回撤(%)": max_dd * 100,
            "Sharpe": sharpe,
            "开仓次数": int(trade_entries),
            "反手次数": int(direction_changes),
            "持仓占比(%)": exposure * 100,
            "胜率(%)": win_rate * 100,
            "盈亏比": pnl_ratio,
        },
    }


def run_elite_tournament(df, fee_rate=0.0006, slippage=0.0002):
    df = calculate_indicators(df)
    strategies = build_strategies(df)

    results = []
    equity_table = pd.DataFrame({"timestamp": df["timestamp"]})

    for name, sig in strategies.items():
        result = backtest_one_strategy(df, sig, fee_rate=fee_rate, slippage=slippage)
        summary = result["summary"]
        summary["选手流派"] = name
        results.append(summary)
        equity_table[name] = result["equity_curve"].values

    report = pd.DataFrame(results)
    report = report[
        [
            "选手流派",
            "净收益(%)",
            "最大回撤(%)",
            "Sharpe",
            "开仓次数",
            "反手次数",
            "持仓占比(%)",
            "胜率(%)",
            "盈亏比",
        ]
    ].sort_values(by="净收益(%)", ascending=False)

    benchmark = ((1 + df["m_ret"]).cumprod().iloc[-1] - 1) * 100

    return report, benchmark, equity_table


def prettify_report(report):
    pretty = report.copy()
    for col in ["净收益(%)", "最大回撤(%)", "持仓占比(%)", "胜率(%)"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:.2f}%")
    pretty["Sharpe"] = pretty["Sharpe"].map(lambda x: f"{x:.2f}")
    pretty["盈亏比"] = pretty["盈亏比"].map(lambda x: f"{x:.2f}")
    return pretty


if __name__ == "__main__":
    df = fetch_safe_5m_data(symbol="BTCUSDT", interval="5m", pages=15)

    if df.empty:
        print("没有成功获取到数据。")
    else:
        report, bench, equity_table = run_elite_tournament(df, fee_rate=0.0006, slippage=0.0002)
        pretty_report = prettify_report(report)

        print(f"\n[{VERSION}] 十大精英策略综合回测")
        print(f"现货大盘基准: {bench:.2f}%")
        print("-" * 90)
        print(pretty_report.to_string(index=False))

        report.to_csv("report.csv", index=False, encoding="utf-8-sig")
        equity_table.to_csv("equity_curves.csv", index=False, encoding="utf-8-sig")
        print("\n已输出 report.csv 与 equity_curves.csv")
