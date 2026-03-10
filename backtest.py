import requests
import pandas as pd
import numpy as np

def fetch_binance_futures_data(symbol="BTCUSDT", interval="4h", limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    df = pd.DataFrame(response.json(), columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def run_strategy(df):
    # 双均线策略
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['signal'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    df['position'] = df['signal'].shift(1) # 防止未来函数作弊
    
    # 计算收益
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['market_returns']
    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略收益'] = (1 + df['strategy_returns']).cumprod() - 1
    return df.dropna()

if __name__ == "__main__":
    df = fetch_binance_futures_data()
    result_df = run_strategy(df)
    
    final_market = result_df['累计市场收益'].iloc[-1] * 100
    final_strategy = result_df['累计策略收益'].iloc[-1] * 100
    
    print(f"回测结束！持有现货收益: {final_market:.2f}% | 策略收益: {final_strategy:.2f}%")
    result_df.to_csv('report.csv', index=False)
