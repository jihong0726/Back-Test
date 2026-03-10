import requests
import pandas as pd
import numpy as np

def fetch_bitget_futures_data(symbol="BTCUSDT", interval="4H", limit=1000):
    print(f"📡 正在拉取 Bitget {symbol} {interval} 级别永续合约数据...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": interval, "limit": limit}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if str(data.get("code")) != "00000":
            return pd.DataFrame()
        kline_list = data.get("data", [])
        if not kline_list:
            return pd.DataFrame()
        df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['close'] = df['close'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

def run_strategy(df):
    if df.empty: return df
    print("🧠 开始计算 [均线趋势 + RSI动能] 复合策略，并扣除真实手续费...")

    # 1. 趋势指标：EMA
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 2. 动能指标：RSI (参数 14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. 复合开仓信号 (强强联合)
    long_cond = (df['ema_20'] > df['ema_50']) & (df['rsi'] > 50)
    short_cond = (df['ema_20'] < df['ema_50']) & (df['rsi'] < 50)

    df['signal'] = 0
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    # 如果没有触发新信号，保持上一根 K 线的持仓状态
    df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

    # 避免未来函数作弊
    df['position'] = df['signal'].shift(1)

    # 4. 引入真实世界的手续费 (Bitget Taker 费率约 0.06%)
    fee_rate = 0.0006
    # 只要当前持仓和上一根K线不一样，说明发生了一笔交易
    df['trade_happened'] = df['position'].diff().fillna(0) != 0
    df['fee_cost'] = np.where(df['trade_happened'], fee_rate, 0)

    # 5. 计算净收益 (市场涨跌幅 * 持仓方向 - 手续费)
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = (df['position'] * df['market_returns']) - df['fee_cost']

    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略收益'] = (1 + df['strategy_returns']).cumprod() - 1

    # 6. 计算最大回撤 (评估风险)
    df['累计资金曲线'] = (1 + df['strategy_returns']).cumprod()
    df['历史最高点'] = df['累计资金曲线'].cummax()
    df['回撤'] = (df['累计资金曲线'] - df['历史最高点']) / df['历史最高点']
    
    return df.dropna()

if __name__ == "__main__":
    df = fetch_bitget_futures_data()
    result_df = run_strategy(df)

    if not result_df.empty:
        final_market = result_df['累计市场收益'].iloc[-1] * 100
        final_strategy = result_df['累计策略收益'].iloc[-1] * 100
        max_drawdown = result_df['回撤'].min() * 100
        trade_count = result_df['trade_happened'].sum()

        print("\n" + "="*50)
        print("📊 优化版：[趋势+动能] 复合策略回测结果")
        print("="*50)
        print(f"一直死拿现货的收益: {final_market:>8.2f}%")
        print(f"优化策略的净收益:   {final_strategy:>8.2f}% (已扣除手续费)")
        print("-" * 50)
        print(f"最大回撤 (风险):    {max_drawdown:>8.2f}%")
        print(f"总交易次数:         {trade_count} 次")
        print("="*50)

        result_df.to_csv('report.csv', index=False)
