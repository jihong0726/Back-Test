import requests
import pandas as pd
import numpy as np
import time

# ==========================================
# ⚙️ 版本控制
# ==========================================
VERSION = "V6.0_Hybrid_Elite_Tactics"

def fetch_safe_5m_data(symbol="BTCUSDT", interval="5m"):
    print(f"[{VERSION}] 📡 抓取 5m 数据并进行多维度清洗...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    end_time = str(int(time.time() * 1000))
    all_data = []
    for i in range(15): # 抓取约 1.5 万根，确保覆盖最近一周多
        params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": interval, "endTime": end_time, "limit": "1000"}
        try:
            res = requests.get(url, params=params).json()
            data = res.get("data", [])
            if not data: break
            new_end_time = data[-1][0]
            if str(new_end_time) == str(end_time): break
            all_data.extend(data)
            end_time = new_end_time
            time.sleep(0.1) 
        except: break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'base_vol']: df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_advanced_indicators(df):
    # 基础均线
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    # 量能过滤：成交量均线
    df['vol_ma'] = df['base_vol'].rolling(20).mean()
    # 波动率：ATR
    df['tr'] = np.maximum((df['high'] - df['low']), 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    # RSI & MACD
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_s'] = df['macd'].ewm(span=9).mean()
    return df

def run_elite_tournament(df):
    df = calculate_advanced_indicators(df)
    fee_rate = 0.0006
    df['m_ret'] = df['close'].pct_change()
    
    strategies = {}
    
    # 1. 强化版卫冕冠军: EMA趋势 + RSI过滤 + 成交量确认
    strategies['1.精英趋势(EMA+RSI+VOL)'] = np.where((df['close'] > df['ema_200']) & (df['rsi'] > 55) & (df['base_vol'] > df['vol_ma']), 1, 
                                            np.where((df['close'] < df['ema_200']) & (df['rsi'] < 45) & (df['base_vol'] > df['vol_ma']), -1, 0))

    # 2. 波动率猎手: 价格突破 EMA + ATR 偏离
    strategies['2.波动率突破(EMA+ATR)'] = np.where(df['close'] > (df['ema_20'] + df['atr']), 1, 
                                           np.where(df['close'] < (df['ema_20'] - df['atr']), -1, 0))

    # 3. 极速动能: MACD金叉 + 强力RSI(>60)
    strategies['3.极速动能(MACD+RSI60)'] = np.where((df['macd'] > df['macd_s']) & (df['rsi'] > 60), 1, 
                                             np.where((df['macd'] < df['macd_s']) & (df['rsi'] < 40), -1, 0))

    # 4. 机构回调: 价格回踩 EMA200 且 RSI超卖
    strategies['4.机构回调(EMA200抄底)'] = np.where((df['close'] > df['ema_200']) & (df['rsi'] < 35), 1, 
                                            np.where((df['close'] < df['ema_200']) & (df['rsi'] > 65), -1, 0))

    # 5. 量能背离: 价格创新高但量能萎缩(逆势)
    strategies['5.量能背离(逆势预警)'] = np.where((df['close'] > df['close'].shift(20)) & (df['base_vol'] < df['vol_ma']), -1, 
                                           np.where((df['close'] < df['close'].shift(20)) & (df['base_vol'] < df['vol_ma']), 1, 0))

    # 6. 均线纠缠突破: EMA20 穿透 EMA200 时的放量
    strategies['6.纠缠突破(EMA20/200)'] = np.where((df['ema_20'] > df['ema_200']) & (df['base_vol'] > df['vol_ma']*1.5), 1, -1)

    # 7. 暴力RSI: 极端 80/20 信号
    strategies['7.暴力RSI(80/20)'] = np.where(df['rsi'] < 20, 1, np.where(df['rsi'] > 80, -1, 0))

    # 8. ATR 止损守卫: 简单的趋势追随但加入 ATR 逻辑
    strategies['8.ATR守卫(趋势追踪)'] = np.where(df['close'] > df['close'].shift(1) + df['atr'], 1, -1)

    # 9. 智能网格: 围绕 EMA20 做高抛低吸
    strategies['9.智能网格(围绕EMA20)'] = np.where(df['close'] < df['ema_20']*0.99, 1, np.where(df['close'] > df['ema_20']*1.01, -1, 0))

    # 10. 终极缝合怪: RSI + MACD + VOL + EMA 全部共振
    strategies['10.全因子共振(四重过滤)'] = np.where((df['rsi']>50) & (df['macd']>0) & (df['base_vol']>df['vol_ma']) & (df['close']>df['ema_200']), 1, -1)

    # 执行回测
    results = []
    total_days = max((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days, 1)
    
    for name, sig in strategies.items():
        pos = pd.Series(sig).replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)
        trades = pos.diff().fillna(0) != 0
        net_ret = (pos * df['m_ret']) - (trades * fee_rate)
        cum_ret = (1 + net_ret).cumprod()
        
        final_p = (cum_ret.iloc[-1] - 1) * 100
        max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100
        
        results.append({"选手流派": name, "净收益(%)": final_p, "最大回撤(%)": f"{max_dd:.2f}%", "交易次数": trades.sum()})
        
    report = pd.DataFrame(results).sort_values(by="净收益(%)", ascending=False)
    report["净收益(%)"] = report["净收益(%)"].apply(lambda x: f"{x:.2f}%")
    return report, ((1 + df['m_ret']).cumprod().iloc[-1] - 1) * 100

if __name__ == "__main__":
    df = fetch_safe_5m_data()
    if not df.empty:
        report, bench = run_elite_tournament(df)
        print(f"\n[{VERSION}] 🏆 十大精英策略综合回测")
        print(f"📈 现货大盘基准: {bench:.2f}%")
        print("-" * 65)
        print(report.to_string(index=False))
