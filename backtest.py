import requests
import pandas as pd
import numpy as np

def fetch_bitget_futures_data(symbol="BTCUSDT", interval="5m", limit=1000):
    print(f"📡 正在拉取 Bitget {symbol} {interval} 级别永续合约数据 (约最近3.5天)...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    # 注意：Bitget 5分钟级别的参数是 '5m'
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
    print("🧠 开始计算 [5m 专属：长效均线 + 严苛 RSI] 复合策略...")

    # 1. 换用更长周期的 EMA 抵抗 5 分钟级别的噪音
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    # 2. RSI 动能指标 (参数 14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. 极度严苛的开仓条件 (必须趋势和动能双重确认)
    long_cond = (df['ema_50'] > df['ema_200']) & (df['rsi'] > 55)
    short_cond = (df['ema_50'] < df['ema_200']) & (df['rsi'] < 45)

    df['signal'] = 0
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

    df['position'] = df['signal'].shift(1)

    # 4. 真实手续费 (Bitget Taker 费率约 0.06%)
    fee_rate = 0.0006
    df['trade_happened'] = df['position'].diff().fillna(0) != 0
    df['fee_cost'] = np.where(df['trade_happened'], fee_rate, 0)

    # 5. 计算毛利与净利润 (直观对比手续费的恐怖)
    df['market_returns'] = df['close'].pct_change()
    
    # 策略毛收益 (没扣手续费)
    df['gross_returns'] = df['position'] * df['market_returns']
    # 策略净收益 (扣了手续费)
    df['net_returns'] = df['gross_returns'] - df['fee_cost']

    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略净收益'] = (1 + df['net_returns']).cumprod() - 1
    df['累计策略毛收益'] = (1 + df['gross_returns']).cumprod() - 1

    df['累计资金曲线'] = (1 + df['net_returns']).cumprod()
    df['历史最高点'] = df['累计资金曲线'].cummax()
    df['回撤'] = (df['累计资金曲线'] - df['历史最高点']) / df['历史最高点']
    
    return df.dropna()

if __name__ == "__main__":
    df = fetch_bitget_futures_data()
    result_df = run_strategy(df)

    if not result_df.empty:
        final_market = result_df['累计市场收益'].iloc[-1] * 100
        final_net = result_df['累计策略净收益'].iloc[-1] * 100
        final_gross = result_df['累计策略毛收益'].iloc[-1] * 100
        max_drawdown = result_df['回撤'].min() * 100
        trade_count = result_df['trade_happened'].sum()

        print("\n" + "="*50)
        print("⚡ 5分钟级别高频回测结果 (最近 3.5 天)")
        print("="*50)
        print(f"死拿现货的收益:     {final_market:>8.2f}%")
        print(f"理想状态的毛收益:   {final_gross:>8.2f}% (如果交易所不收手续费)")
        print(f"扣除手续费后净赚:   {final_net:>8.2f}% (现实结果)")
        print("-" * 50)
        print(f"最大回撤 (风险):    {max_drawdown:>8.2f}%")
        print(f"总交易次数:         {trade_count} 次")
        print("="*50)

        result_df.to_csv('report.csv', index=False)
