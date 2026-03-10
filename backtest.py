import requests
import pandas as pd
import numpy as np
import time

# ==========================================
# ⚙️ 版本控制
# ==========================================
VERSION = "V5.0_5m_HighFreq_10_Fighters"

def fetch_safe_5m_data(symbol="BTCUSDT", interval="5m"):
    print(f"[{VERSION}] 📡 启动 5 分钟级别数据抓取 (自动探测交易所底线)...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    end_time = str(int(time.time() * 1000))
    
    all_data = []
    # 设置合理上限，防止鬼打墙
    for i in range(100): 
        params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": interval, "endTime": end_time, "limit": "1000"}
        try:
            res = requests.get(url, params=params).json()
            data = res.get("data", [])
            
            if not data:
                break
                
            new_end_time = data[-1][0]
            # 修复“鬼打墙”漏洞：如果交易所返回的最老时间不再推进，直接跳出
            if str(new_end_time) == str(end_time):
                print("🛑 触及交易所 5 分钟数据历史深度墙，停止抓取。")
                break
                
            all_data.extend(data)
            end_time = new_end_time
            time.sleep(0.15) 
        except Exception as e:
            print(f"抓取中断: {e}")
            break

    # 清洗和去重
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.drop_duplicates(subset=['timestamp']) # 确保没有重复的脏数据
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"✅ 成功获取 {len(df)} 根【真实有效】的 5 分钟 K线！")
    return df

def calculate_all_indicators(df):
    # 均线系统
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['sma_20'] + 2 * df['bb_std']
    df['bb_lower'] = df['sma_20'] - 2 * df['bb_std']
    
    # MACD
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 海龟突破
    df['high_20'] = df['high'].rolling(20).max().shift(1)
    df['low_20'] = df['low'].rolling(20).min().shift(1)
    df['high_50'] = df['high'].rolling(50).max().shift(1)
    df['low_50'] = df['low'].rolling(50).min().shift(1)
    
    # ATR
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['atr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1).rolling(14).mean()

    return df

def battle_royale_5m(df):
    df = calculate_all_indicators(df)
    fee_rate = 0.0006
    df['market_returns'] = df['close'].pct_change()
    
    strategies = {}
    
    # 10 位选手的核心逻辑（不变，但在5分钟线上触发会极其频繁）
    strategies['1.老军医(SMA50/200)'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    strategies['2.快刀手(EMA10/30)'] = np.where(df['ema_10'] > df['ema_30'], 1, -1)
    
    sig3 = pd.Series(0, index=df.index)
    sig3.loc[(df['ema_50'] > df['sma_200']) & (df['rsi'] > 55)] = 1
    sig3.loc[(df['ema_50'] < df['sma_200']) & (df['rsi'] < 45)] = -1
    strategies['3.卫冕冠军(均线+RSI)'] = sig3.replace(0, np.nan).ffill().fillna(0)
    
    sig4 = pd.Series(np.nan, index=df.index)
    sig4.loc[df['close'] < df['bb_lower']] = 1
    sig4.loc[df['close'] > df['bb_upper']] = -1
    strategies['4.抄底狂魔(布林带逆势)'] = sig4.ffill().fillna(0)
    
    strategies['5.动能骑士(MACD柱子)'] = np.where(df['macd_hist'] > 0, 1, -1)
    
    sig6 = pd.Series(np.nan, index=df.index)
    sig6.loc[df['close'] > df['high_20']] = 1
    sig6.loc[df['close'] < df['low_20']] = -1
    strategies['6.小海龟(20突破)'] = sig6.ffill().fillna(0)
    
    sig7 = pd.Series(np.nan, index=df.index)
    sig7.loc[df['close'] > df['high_50']] = 1
    sig7.loc[df['close'] < df['low_50']] = -1
    strategies['7.大海龟(50突破)'] = sig7.ffill().fillna(0)
    
    sig8 = pd.Series(np.nan, index=df.index)
    sig8.loc[df['rsi'] < 30] = 1
    sig8.loc[df['rsi'] > 70] = -1
    strategies['8.极端反转(纯RSI)'] = sig8.ffill().fillna(0)
    
    sig9 = pd.Series(0, index=df.index)
    sig9.loc[(df['close'] > df['ema_200']) & (df['close'] < df['sma_20'])] = 1
    sig9.loc[(df['close'] < df['ema_200']) & (df['close'] > df['sma_20'])] = -1
    strategies['9.趋势回调(牛市买跌)'] = sig9.replace(0, np.nan).ffill().fillna(0)
    
    sig10 = pd.Series(np.nan, index=df.index)
    sig10.loc[df['close'] > (df['sma_50'] + df['atr'])] = 1
    sig10.loc[df['close'] < (df['sma_50'] - df['atr'])] = -1
    strategies['10.波动率猎手(ATR突破)'] = sig10.ffill().fillna(0)

    results = []
    total_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    total_days = total_days if total_days > 0 else 1

    for name, sig_array in strategies.items():
        pos = pd.Series(sig_array).shift(1).fillna(0) 
        
        trade_happened = pos.diff().fillna(0) != 0
        fee_cost = np.where(trade_happened, fee_rate, 0)
        
        net_returns = (pos * df['market_returns']) - fee_cost
        cum_returns = (1 + net_returns).cumprod()
        
        cum_max = cum_returns.cummax()
        drawdown = (cum_returns - cum_max) / cum_max
        
        final_return = (cum_returns.iloc[-1] - 1) * 100
        max_dd = drawdown.min() * 100
        trades = trade_happened.sum()
        
        results.append({
            "选手编号与流派": name,
            "净收益(%)": final_return, 
            "最大回撤(%)": f"{max_dd:.2f}%",
            "交易次数": trades,
            "日均交易": f"{trades / total_days:.1f}次"
        })
        
    report_df = pd.DataFrame(results).sort_values(by="净收益(%)", ascending=False)
    report_df["净收益(%)"] = report_df["净收益(%)"].apply(lambda x: f"{x:.2f}%")
    
    benchmark_return = ((1 + df['market_returns']).cumprod().iloc[-1] - 1) * 100
    
    return report_df, benchmark_return, df

if __name__ == "__main__":
    df = fetch_safe_5m_data()
    
    if not df.empty:
        report_df, benchmark, raw_df = battle_royale_5m(df)
        
        print("\n" + "⚔️"*30)
        print(f"🥊 5分钟高频死亡擂台 (10大门派) | 版本: {VERSION}")
        print("⚔️"*30)
        print(f"⏱️ 极限回测区间: {raw_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} 至 {raw_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📉 现货大盘基准收益: {benchmark:.2f}%")
        print("-" * 60)
        print(report_df.to_string(index=False))
        print("⚔️"*30)
        
        raw_df.to_csv('report.csv', index=False)
