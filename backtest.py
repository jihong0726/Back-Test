import requests
import pandas as pd
import numpy as np

def fetch_bitget_futures_data(symbol="BTCUSDT", interval="4H", limit=1000):
    print(f"📡 正在拉取 Bitget {symbol} {interval} 级别永续合约数据...")
    
    # 使用 Bitget V2 版本的公共合约 K 线接口
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES", # 明确指定为 USDT 本位永续合约
        "granularity": interval,       # K线级别 (Bitget 使用 1H, 4H, 1D)
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Bitget 的成功状态码是 "00000"
        if str(data.get("code")) != "00000":
            print(f"❌ 交易所 API 返回错误: {data.get('msg')}")
            return pd.DataFrame()
            
        # Bitget 返回的 K 线数组格式: [时间戳, 开盘价, 最高价, 最低价, 收盘价, 基础币交易量, 计价币交易量]
        kline_list = data.get("data", [])
        if not kline_list:
            print("⚠️ API 请求成功，但返回的数据为空。")
            return pd.DataFrame()

        df = pd.DataFrame(kline_list, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'
        ])
        
        # ⚠️ 关键安全机制：确保数据是按时间正序排列的（从旧到新），否则均线计算会完全反过来！
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 数据类型转换
        df['close'] = df['close'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df

    except Exception as e:
        print(f"❌ 网络请求发生严重异常: {e}")
        return pd.DataFrame()

def run_strategy(df):
    if df.empty:
        return df
        
    print("🧠 开始计算双均线交叉策略 (EMA20 vs EMA50)...")
    
    # 1. 计算技术指标
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 2. 生成交易信号 (1 = 持有多单, -1 = 持有空单)
    df['signal'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    
    # 3. 避免未来函数作弊（今天的信号，明天才能基于开盘价执行）
    df['position'] = df['signal'].shift(1)
    
    # 4. 计算收益
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['market_returns']
    
    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略收益'] = (1 + df['strategy_returns']).cumprod() - 1
    
    return df.dropna()

if __name__ == "__main__":
    # 调用 Bitget 专属的数据抓取函数
    df = fetch_bitget_futures_data()
    result_df = run_strategy(df)
    
    if result_df.empty:
        print("\n⚠️ 数据处理失败，回测中止。")
    else:
        final_market = result_df['累计市场收益'].iloc[-1] * 100
        final_strategy = result_df['累计策略收益'].iloc[-1] * 100
        
        print("\n" + "="*45)
        print("📊 Bitget 永续合约回测结果 (去除手续费前)")
        print("="*45)
        print(f"交易标的: BTC-USDT 永续")
        print(f"回测周期: {result_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} 至 {result_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print("-" * 45)
        print(f"一直死拿比特币的收益: {final_market:>8.2f}%")
        print(f"双均线策略的总收益:   {final_strategy:>8.2f}%")
        print("="*45)
        
        # 将结果输出到 CSV 供下载分析
        result_df.to_csv('report.csv', index=False)
        print("📄 详细明细已保存至 report.csv")
