import os
import json
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata

# ==========================================
# ⚙️ 版本控制
# ==========================================
VERSION = "V7.0_多因子代理回测框架"
REPORT_DIR = "reports"

# ==========================================
# 🔧 显示工具：处理中英文宽度，保证对齐
# ==========================================
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

# ==========================================
# 🌐 数据抓取
# ==========================================
def fetch_candles_bitget(symbol="BTCUSDT", interval="5m", rounds=15, limit=1000):
    print(f"[{VERSION}] 抓取 {symbol} {interval} K线数据...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    end_time = str(int(time.time() * 1000))
    all_data = []

    for _ in range(rounds):
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "endTime": end_time,
            "limit": str(limit)
        }

        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json().get("data", [])
            if not data:
                break

            new_end_time = data[-1][0]
            if str(new_end_time) == str(end_time):
                break

            all_data.extend(data)
            end_time = new_end_time
            time.sleep(0.08)
        except Exception as e:
            print(f"抓取失败: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data,
        columns=["timestamp", "open", "high", "low", "close", "base_vol", "quote_vol"]
    )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "base_vol", "quote_vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def fetch_open_interest(symbol="BTCUSDT"):
    try:
        url = "https://api.bitget.com/api/v2/mix/market/open-interest"
        params = {"symbol": symbol, "productType": "USDT-FUTURES"}
        res = requests.get(url, params=params, timeout=10).json()
        data = res.get("data", {})
        oi = float(data.get("size", 0))
        return oi
    except Exception:
        return np.nan

def fetch_funding_rate(symbol="BTCUSDT"):
    try:
        url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
        params = {"symbol": symbol, "productType": "USDT-FUTURES"}
        res = requests.get(url, params=params, timeout=10).json()
        data = res.get("data", {})
        fr = float(data.get("fundingRate", 0))
        return fr
    except Exception:
        return np.nan

# ==========================================
# 📐 技术指标
# ==========================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calc_atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return tr, atr

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["vol_ma_20"] = df["base_vol"].rolling(20).mean()
    df["vol_ma_50"] = df["base_vol"].rolling(50).mean()

    df["rsi_7"] = calc_rsi(df["close"], 7)
    df["rsi_14"] = calc_rsi(df["close"], 14)

    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["close"])
    df["tr"], df["atr_14"] = calc_atr(df, 14)
    df["atr_3"] = df["tr"].rolling(3).mean()

    df["ret"] = df["close"].pct_change().fillna(0)

    return df

def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp = temp.set_index("timestamp")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "base_vol": "sum",
        "quote_vol": "sum"
    }
    df4h = temp.resample("4H").agg(agg).dropna().reset_index()
    df4h = add_indicators(df4h)
    return df4h

# ==========================================
# 🧠 生成代理 Prompt 快照数据
# ==========================================
def build_agent_snapshot(df_5m: pd.DataFrame, df_4h: pd.DataFrame, symbol="BTCUSDT"):
    latest = df_5m.iloc[-1]
    last10 = df_5m.tail(10)
    latest_4h = df_4h.iloc[-1]
    last10_4h = df_4h.tail(10)

    oi_latest = fetch_open_interest(symbol)
    funding_rate = fetch_funding_rate(symbol)

    snapshot = {
        "symbol": symbol,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_market_state": {
            "current_price": round(float(latest["close"]), 6),
            "current_ema20": round(float(latest["ema_20"]), 6),
            "current_macd": round(float(latest["macd"]), 6),
            "current_rsi_7": round(float(latest["rsi_7"]), 6),
            "open_interest_latest": None if pd.isna(oi_latest) else round(float(oi_latest), 6),
            "open_interest_average_est": None if pd.isna(oi_latest) else round(float(oi_latest), 6),
            "funding_rate": None if pd.isna(funding_rate) else round(float(funding_rate), 8),
            "intraday_series": {
                "mid_prices": [round(float(x), 6) for x in last10["close"].tolist()],
                "ema20": [round(float(x), 6) for x in last10["ema_20"].tolist()],
                "macd": [round(float(x), 6) for x in last10["macd"].tolist()],
                "rsi_7": [round(float(x), 6) for x in last10["rsi_7"].tolist()],
                "rsi_14": [round(float(x), 6) for x in last10["rsi_14"].tolist()]
            },
            "longer_term_context_4h": {
                "ema20": round(float(latest_4h["ema_20"]), 6),
                "ema50": round(float(latest_4h["ema_50"]), 6),
                "atr_3": round(float(latest_4h["atr_3"]), 6),
                "atr_14": round(float(latest_4h["atr_14"]), 6),
                "current_volume": round(float(latest_4h["base_vol"]), 6),
                "average_volume_20": round(float(df_4h["base_vol"].tail(20).mean()), 6),
                "macd_series": [round(float(x), 6) for x in last10_4h["macd"].tolist()],
                "rsi_14_series": [round(float(x), 6) for x in last10_4h["rsi_14"].tolist()]
            }
        }
    }
    return snapshot

# ==========================================
# 📊 交易策略集合
# ==========================================
def generate_strategies(df: pd.DataFrame):
    s = {}

    # 趋势
    s["01.趋势跟随_EMA20x50_RSI"] = np.where(
        (df["ema_20"] > df["ema_50"]) & (df["rsi_14"] > 55),
        1,
        np.where((df["ema_20"] < df["ema_50"]) & (df["rsi_14"] < 45), -1, 0)
    )

    s["02.趋势跟随_EMA20x200_RSI"] = np.where(
        (df["close"] > df["ema_200"]) & (df["rsi_14"] > 52),
        1,
        np.where((df["close"] < df["ema_200"]) & (df["rsi_14"] < 48), -1, 0)
    )

    s["03.MACD金叉_RSI过滤"] = np.where(
        (df["macd"] > df["macd_signal"]) & (df["rsi_7"] > 58),
        1,
        np.where((df["macd"] < df["macd_signal"]) & (df["rsi_7"] < 42), -1, 0)
    )

    s["04.MACD柱体翻正_趋势确认"] = np.where(
        (df["macd_hist"] > 0) & (df["close"] > df["ema_20"]),
        1,
        np.where((df["macd_hist"] < 0) & (df["close"] < df["ema_20"]), -1, 0)
    )

    # 放量
    s["05.放量突破_EMA20_VOL"] = np.where(
        (df["close"] > df["ema_20"]) & (df["base_vol"] > df["vol_ma_20"] * 1.2),
        1,
        np.where((df["close"] < df["ema_20"]) & (df["base_vol"] > df["vol_ma_20"] * 1.2), -1, 0)
    )

    s["06.放量趋势_EMA50_VOL"] = np.where(
        (df["close"] > df["ema_50"]) & (df["base_vol"] > df["vol_ma_50"] * 1.1),
        1,
        np.where((df["close"] < df["ema_50"]) & (df["base_vol"] > df["vol_ma_50"] * 1.1), -1, 0)
    )

    # ATR / 波动
    s["07.ATR突破_EMA20"] = np.where(
        df["close"] > (df["ema_20"] + df["atr_14"]),
        1,
        np.where(df["close"] < (df["ema_20"] - df["atr_14"]), -1, 0)
    )

    s["08.ATR突破_EMA50"] = np.where(
        df["close"] > (df["ema_50"] + df["atr_14"] * 0.8),
        1,
        np.where(df["close"] < (df["ema_50"] - df["atr_14"] * 0.8), -1, 0)
    )

    # 均值回归
    s["09.RSI超卖超买_均值回归"] = np.where(
        df["rsi_7"] < 25,
        1,
        np.where(df["rsi_7"] > 75, -1, 0)
    )

    s["10.偏离EMA20_网格反转"] = np.where(
        df["close"] < df["ema_20"] * 0.985,
        1,
        np.where(df["close"] > df["ema_20"] * 1.015, -1, 0)
    )

    s["11.偏离EMA50_网格反转"] = np.where(
        df["close"] < df["ema_50"] * 0.98,
        1,
        np.where(df["close"] > df["ema_50"] * 1.02, -1, 0)
    )

    # 多因子组合
    s["12.四因子共振_EMA_MACD_RSI_VOL"] = np.where(
        (df["close"] > df["ema_200"]) &
        (df["macd"] > 0) &
        (df["rsi_14"] > 52) &
        (df["base_vol"] > df["vol_ma_20"]),
        1,
        np.where(
            (df["close"] < df["ema_200"]) &
            (df["macd"] < 0) &
            (df["rsi_14"] < 48) &
            (df["base_vol"] > df["vol_ma_20"]),
            -1,
            0
        )
    )

    s["13.强动能共振_EMA9_20_MACD_RSI7"] = np.where(
        (df["ema_9"] > df["ema_20"]) &
        (df["macd_hist"] > 0) &
        (df["rsi_7"] > 60),
        1,
        np.where(
            (df["ema_9"] < df["ema_20"]) &
            (df["macd_hist"] < 0) &
            (df["rsi_7"] < 40),
            -1,
            0
        )
    )

    s["14.保守趋势_EMA200_RSI14_VOL"] = np.where(
        (df["close"] > df["ema_200"]) &
        (df["rsi_14"] > 55) &
        (df["base_vol"] > df["vol_ma_20"] * 1.1),
        1,
        np.where(
            (df["close"] < df["ema_200"]) &
            (df["rsi_14"] < 45) &
            (df["base_vol"] > df["vol_ma_20"] * 1.1),
            -1,
            0
        )
    )

    # 反转
    s["15.低位反弹_RSI7_MACD修复"] = np.where(
        (df["rsi_7"] < 30) & (df["macd_hist"] > df["macd_hist"].shift(1)),
        1,
        np.where((df["rsi_7"] > 70) & (df["macd_hist"] < df["macd_hist"].shift(1)), -1, 0)
    )

    s["16.高低点突破_20周期"] = np.where(
        df["close"] > df["high"].shift(1).rolling(20).max(),
        1,
        np.where(df["close"] < df["low"].shift(1).rolling(20).min(), -1, 0)
    )

    # 混合型组合
    s["17.趋势+突破_EMA50_ATR_MACD"] = np.where(
        (df["close"] > df["ema_50"]) &
        (df["close"] > df["ema_20"] + df["atr_14"] * 0.5) &
        (df["macd"] > df["macd_signal"]),
        1,
        np.where(
            (df["close"] < df["ema_50"]) &
            (df["close"] < df["ema_20"] - df["atr_14"] * 0.5) &
            (df["macd"] < df["macd_signal"]),
            -1,
            0
        )
    )

    s["18.趋势+量能_EMA200_RSI_VOL加强"] = np.where(
        (df["close"] > df["ema_200"]) &
        (df["rsi_14"] > 58) &
        (df["base_vol"] > df["vol_ma_20"] * 1.3),
        1,
        np.where(
            (df["close"] < df["ema_200"]) &
            (df["rsi_14"] < 42) &
            (df["base_vol"] > df["vol_ma_20"] * 1.3),
            -1,
            0
        )
    )

    s["19.短线强势_EMA9_MACD_RSI7_VOL"] = np.where(
        (df["close"] > df["ema_9"]) &
        (df["macd_hist"] > 0) &
        (df["rsi_7"] > 62) &
        (df["base_vol"] > df["vol_ma_20"]),
        1,
        np.where(
            (df["close"] < df["ema_9"]) &
            (df["macd_hist"] < 0) &
            (df["rsi_7"] < 38) &
            (df["base_vol"] > df["vol_ma_20"]),
            -1,
            0
        )
    )

    s["20.极端反转_RSI7_20_80"] = np.where(
        df["rsi_7"] < 20,
        1,
        np.where(df["rsi_7"] > 80, -1, 0)
    )

    s["21.MACD零轴趋势"] = np.where(
        (df["macd"] > 0) & (df["close"] > df["ema_20"]),
        1,
        np.where((df["macd"] < 0) & (df["close"] < df["ema_20"]), -1, 0)
    )

    s["22.价格站上双均线"] = np.where(
        (df["close"] > df["ema_20"]) & (df["close"] > df["ema_50"]),
        1,
        np.where((df["close"] < df["ema_20"]) & (df["close"] < df["ema_50"]), -1, 0)
    )

    s["23.RSI14中轴突破"] = np.where(
        df["rsi_14"] > 55,
        1,
        np.where(df["rsi_14"] < 45, -1, 0)
    )

    s["24.量价同步突破"] = np.where(
        (df["close"] > df["close"].shift(5).rolling(10).max()) &
        (df["base_vol"] > df["vol_ma_20"] * 1.25),
        1,
        np.where(
            (df["close"] < df["close"].shift(5).rolling(10).min()) &
            (df["base_vol"] > df["vol_ma_20"] * 1.25),
            -1,
            0
        )
    )

    s["25.全因子保守版"] = np.where(
        (df["close"] > df["ema_200"]) &
        (df["ema_20"] > df["ema_50"]) &
        (df["macd_hist"] > 0) &
        (df["rsi_14"] > 55) &
        (df["base_vol"] > df["vol_ma_20"]),
        1,
        np.where(
            (df["close"] < df["ema_200"]) &
            (df["ema_20"] < df["ema_50"]) &
            (df["macd_hist"] < 0) &
            (df["rsi_14"] < 45) &
            (df["base_vol"] > df["vol_ma_20"]),
            -1,
            0
        )
    )

    return s

# ==========================================
# 🧪 回测核心
# ==========================================
def run_backtest_for_signal(df: pd.DataFrame, signal: pd.Series, fee_rate=0.0006):
    pos = pd.Series(signal, index=df.index)
    pos = pos.replace(0, np.nan).ffill().fillna(0)
    pos = pos.shift(1).fillna(0)

    trades = pos.diff().fillna(0).abs()
    gross_ret = pos * df["ret"]
    fee_cost = trades * fee_rate
    net_ret = gross_ret - fee_cost

    equity = (1 + net_ret).cumprod()

    total_return = (equity.iloc[-1] - 1) * 100
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min() * 100

    wins = (net_ret > 0).sum()
    losses = (net_ret < 0).sum()
    total_bars = len(net_ret[net_ret != 0])
    win_rate = (wins / total_bars * 100) if total_bars > 0 else 0

    avg_ret = net_ret.mean()
    std_ret = net_ret.std()
    sharpe = (avg_ret / std_ret * math.sqrt(252 * 24 * 12)) if std_ret and not np.isnan(std_ret) and std_ret != 0 else 0

    return {
        "净收益(%)": total_return,
        "最大回撤(%)": max_dd,
        "信号变更次数": int(trades.sum()),
        "胜率(%)": win_rate,
        "Sharpe": sharpe,
        "最终资金曲线": equity
    }

def run_strategy_tournament(df: pd.DataFrame):
    strategies = generate_strategies(df)
    results = []
    curves = {}

    for name, sig in strategies.items():
        r = run_backtest_for_signal(df, pd.Series(sig))
        curves[name] = r["最终资金曲线"]
        results.append({
            "策略名称": name,
            "净收益(%)": f'{r["净收益(%)"]:.2f}%',
            "最大回撤(%)": f'{r["最大回撤(%)"]:.2f}%',
            "胜率(%)": f'{r["胜率(%)"]:.2f}%',
            "Sharpe": f'{r["Sharpe"]:.2f}',
            "信号变更次数": r["信号变更次数"]
        })

    report = pd.DataFrame(results)

    report["_净收益排序"] = report["净收益(%)"].str.replace("%", "", regex=False).astype(float)
    report["_回撤排序"] = report["最大回撤(%)"].str.replace("%", "", regex=False).astype(float)
    report["_Sharpe排序"] = report["Sharpe"].astype(float)

    report = report.sort_values(
        by=["_净收益排序", "_Sharpe排序"],
        ascending=[False, False]
    ).reset_index(drop=True)

    report = report.drop(columns=["_净收益排序", "_回撤排序", "_Sharpe排序"])
    return report, curves

# ==========================================
# 💾 输出
# ==========================================
def ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)

def save_outputs(report: pd.DataFrame, snapshot: dict, symbol: str):
    ensure_report_dir()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(REPORT_DIR, f"{symbol}_策略回测报告_{now_str}.csv")
    json_path = os.path.join(REPORT_DIR, f"{symbol}_代理快照_{now_str}.json")
    txt_path = os.path.join(REPORT_DIR, f"{symbol}_策略回测报告_{now_str}.txt")

    report.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    table_text = format_table(
        report,
        right_align_cols={"净收益(%)", "最大回撤(%)", "胜率(%)", "Sharpe", "信号变更次数"}
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    return csv_path, json_path, txt_path

# ==========================================
# 🚀 主程序
# ==========================================
def main():
    symbol = "BTCUSDT"
    interval = "5m"

    df = fetch_candles_bitget(symbol=symbol, interval=interval, rounds=15, limit=1000)
    if df.empty:
        print("抓不到数据，程序结束。")
        return

    df = add_indicators(df)
    df4h = resample_4h(df)

    # 跑回测
    report, curves = run_strategy_tournament(df)

    # 基准
    benchmark = ((df["close"].iloc[-1] / df["close"].iloc[0]) - 1) * 100

    # 构建代理快照
    snapshot = build_agent_snapshot(df, df4h, symbol=symbol)

    # 输出文件
    csv_path, json_path, txt_path = save_outputs(report, snapshot, symbol)

    # 控制台输出
    print(f"\n[{VERSION}] 多策略综合回测结果")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易标的: {symbol}")
    print(f"周期: {interval}")
    print(f"样本K线数: {len(df)}")
    print(f"现货基准涨跌: {benchmark:.2f}%")
    print("=" * 110)

    top10 = report.head(10)
    table_text = format_table(
        top10,
        right_align_cols={"净收益(%)", "最大回撤(%)", "胜率(%)", "Sharpe", "信号变更次数"}
    )
    print(table_text)

    print("\n文件已输出:")
    print(f"- CSV 报告: {csv_path}")
    print(f"- JSON 快照: {json_path}")
    print(f"- TXT 对齐表: {txt_path}")

if __name__ == "__main__":
    main()
