import requests
import pandas as pd
import numpy as np
import time

# ==========================================
# ⚙️ 版本控制
# ==========================================
VERSION = "V4.0_High_Frequency_Daily_Trader"

def fetch_massive_5m_data(symbol="BTCUSDT", interval="5m"):
    print(f"[{VERSION}] 📡 正在挑战交易所极限：疯狂拉取 5分钟 历史数据...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    end_time = str(int(time.time() * 1000))
    # 试图拉取 2 年前的数据
    target_time = int((time.time() - 2 * 365 * 24 * 60 * 60) * 1000) 
    
    all_data = []
    # 2年5分钟线大约需要 210 次请求 (每次 1000 根)
    # 我们设置最大循环 250 次，但如果交易所中途不给数据了，会自动停止
    for i in range(250): 
        params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": interval, "endTime": end_time, "limit": "1000"}
        try:
            res = requests.get(url, params=params).json()
            data = res.get("data", [])
            
            # 交易所 API 的隐藏墙：如果不给数据了，说明触及了历史深度限制
            if not data:
                print(f"⚠️ 交易所公共 API 历史深度触达极限！已无法获取更早的数据。")
                break
                
            all_data.extend(data)
            end_time = data[-1][0]
            
            if int(end_time) < target_time: 
                break
                
            # 打印进度条，安抚焦虑
            if i % 10 == 0:
                print(f"   已抓取 {len(all_data)} 根 K 线，当前追溯至: {pd.to_datetime(int(end_time), unit='ms').strftime('%Y-%m-%d')}")
                
            time.sleep(0.15) # 防止触发 429 Too Many Requests
        except Exception as e:
            print(f"抓取中断: {e}")
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✅ 最终成功获取了 {len(df)} 根 5分钟 K线！起始时间: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')}")
    return df

def run_high_frequency_strategy(df):
    if df.empty: return df
    
    # 极度敏感的指标：EMA 5 和 EMA 15，保证每天频繁交叉
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    
    df['signal'] = np.where(df['ema_5'] > df['ema_15'], 1, -1)
    df['position'] = df['signal'].shift(1).fillna(0)

    # 计算真实利润与 0.06% 手续费摩擦
    fee_rate = 0.0006
    df['market_returns'] = df['close'].pct_change()
    df['trade_happened'] = df['position'].diff().fillna(0) != 0
    df['fee_cost'] = np.where(df['trade_happened'], fee_rate, 0)
    
    df['gross_returns'] = df['position'] * df['market_returns']
    df['net_returns'] = df['gross_returns'] - df['fee_cost']
    
    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略毛收益'] = (1 + df['gross_returns']).cumprod() - 1
    df['累计策略净收益'] = (1 + df['net_returns']).cumprod() - 1
    
    # 计算日均交易次数，验证是否达标
    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    total_days = total_days if total_days > 0 else 1
    total_trades = df['trade_happened'].sum()
    trades_per_day = total_trades / total_days

    return df, total_trades, trades_per_day

if __name__ == "__main__":
    df = fetch_massive_5m_data()
    
    if not df.empty:
        res_df, total_trades, trades_per_day = run_high_frequency_strategy(df)
        
        final_market = res_df['累计市场收益'].iloc[-1] * 100
        final_gross = res_df['累计策略毛收益'].iloc[-1] * 100
        final_net = res_df['累计策略净收益'].iloc[-1] * 100
        
        print("\n" + "⚔️"*25)
        print(f"🌪️ 高频日内策略测评报告 (5m级别)")
        print("⚔️"*25)
        print(f"回测时间跨度: {res_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} 至 {res_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"总交易次数:   {total_trades} 次")
        print(f"平均每日交易: {trades_per_day:.1f} 次 (强制每日开仓达标！)")
        print("-" * 50)
        print(f"基准表现(死拿):     {final_market:>9.2f}%")
        print(f"理想状态(不扣手续费): {final_gross:>9.2f}%")
        print(f"现实结果(扣除手续费): {final_net:>9.2f}%")
        print("⚔️"*25)
        print("💡 终极思考：对比一下理想状态和现实结果的巨大落差，你现在感受到高频交易里手续费的恐怖了吗？")
        
        res_df.to_csv('report.csv', index=False)
