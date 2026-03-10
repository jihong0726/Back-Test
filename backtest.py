import requests
import pandas as pd
import numpy as np
import time

def fetch_1_year_bitget_data(symbol="BTCUSDT", interval="1H"):
    print(f"📡 启动时光机：自动循环拉取 Bitget {symbol} {interval} 级别 1 年历史数据...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    
    # 获取当前时间戳和1年前的时间戳
    end_time = str(int(time.time() * 1000))
    one_year_ago = int((time.time() - 365 * 24 * 60 * 60) * 1000)
    
    all_data = []
    
    # 循环分页抓取 (每次 1000 根，1年 1H 大约需要抓 9 次)
    for i in range(12): 
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "endTime": end_time,
            "limit": "1000"
        }
        try:
            res = requests.get(url, params=params).json()
            data = res.get("data", [])
            if not data:
                break
                
            all_data.extend(data)
            end_time = data[-1][0] # 把下一轮的结束时间设为当前这批最老的数据时间
            
            # 如果已经抓到了1年前的数据，提前结束循环
            if int(end_time) < one_year_ago:
                break
                
            time.sleep(0.2) # 礼貌延迟，防止被封 IP
        except Exception as e:
            print(f"抓取中断: {e}")
            break

    # 清洗组合数据
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 严格截取刚好 1 年的数据
    df = df[df['timestamp'] >= one_year_ago].copy()
    
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    print(f"✅ 成功拼接 {len(df)} 根 K 线，跨度整整 365 天！\n")
    return df

def calculate_technical_indicators(df):
    # 为 5 个策略准备所需的全部技术指标
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def run_5_strategies(df):
    df = calculate_technical_indicators(df)
    
    # 定义 5 个策略的信号 (1=多, -1=空)
    
    # 策略 A: 短线均线 (EMA 20/50)
    df['sig_A'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    
    # 策略 B: 长线均线 (SMA 50/200)
    df['sig_B'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    
    # 策略 C: 趋势+动能 (EMA 50/200 + RSI)
    df['sig_C'] = 0
    df.loc[(df['ema_50'] > df['sma_200']) & (df['rsi'] > 55), 'sig_C'] = 1
    df.loc[(df['ema_50'] < df['sma_200']) & (df['rsi'] < 45), 'sig_C'] = -1
    df['sig_C'] = df['sig_C'].replace(0, np.nan).ffill().fillna(0)
    
    # 策略 D: MACD 交叉
    df['sig_D'] = np.where(df['macd'] > df['signal_line'], 1, -1)
    
    # 策略 E: 布林带逆势 (突破下轨做多，突破上轨做空)
    df['sig_E'] = np.nan
    df.loc[df['close'] < df['bb_lower'], 'sig_E'] = 1
    df.loc[df['close'] > df['bb_upper'], 'sig_E'] = -1
    df['sig_E'] = df['sig_E'].ffill().fillna(0)

    # 统一计算收益与摩擦成本
    strategies = ['A', 'B', 'C', 'D', 'E']
    names = ['EMA快线', 'SMA慢线', '均线+RSI', 'MACD趋势', '布林带逆势']
    fee_rate = 0.0006
    df['market_returns'] = df['close'].pct_change()
    
    results = []
    
    for sig, name in zip(strategies, names):
        col_sig = f'sig_{sig}'
        col_pos = f'pos_{sig}'
        
        # 移位避免未来函数
        df[col_pos] = df[col_sig].shift(1)
        
        # 计算手续费
        trade_happened = df[col_pos].diff().fillna(0) != 0
        fee_cost = np.where(trade_happened, fee_rate, 0)
        
        # 计算净收益并累计
        net_returns = (df[col_pos] * df['market_returns']) - fee_cost
        cum_returns = (1 + net_returns).cumprod()
        
        # 计算最大回撤
        cum_max = cum_returns.cummax()
        drawdown = (cum_returns - cum_max) / cum_max
        
        final_return = (cum_returns.iloc[-1] - 1) * 100
        max_dd = drawdown.min() * 100
        trades = trade_happened.sum()
        
        results.append({
            "策略": name,
            "净收益(%)": f"{final_return:.2f}%",
            "最大回撤(%)": f"{max_dd:.2f}%",
            "交易次数": trades
        })
        
    # 计算死拿现货的基准收益
    benchmark_return = ((1 + df['market_returns']).cumprod().iloc[-1] - 1) * 100
    
    return pd.DataFrame(results), benchmark_return, df

if __name__ == "__main__":
    df = fetch_1_year_bitget_data()
    
    if not df.empty:
        report_df, benchmark, raw_df = run_5_strategies(df)
        
        print("="*60)
        print(f"🏆 1年期多因子策略争霸赛 (扣除 0.06% 真实手续费)")
        print(f"回测区间: {raw_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} 至 {raw_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"死拿现货的基准收益: {benchmark:.2f}%")
        print("="*60)
        # 漂亮地打印出对比表格
        print(report_df.to_string(index=False))
        print("="*60)
        print("💡 分析提示：")
        print("1. 比较不同策略的【交易次数】，感受长线与短线的摩擦成本差异。")
        print("2. 在熊市/牛市交替的 1 年里，看看逆势策略(布林带)和趋势策略谁活得更好。")
        
        raw_df.to_csv('report.csv', index=False)
