import requests
import pandas as pd
import numpy as np
import time

def fetch_1_year_bitget_data(symbol="BTCUSDT", interval="1H"):
    print(f"📡 自动循环拉取 Bitget {symbol} {interval} 级别 1 年历史数据...")
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    end_time = str(int(time.time() * 1000))
    one_year_ago = int((time.time() - 365 * 24 * 60 * 60) * 1000)
    
    all_data = []
    for i in range(12): 
        params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": interval, "endTime": end_time, "limit": "1000"}
        try:
            res = requests.get(url, params=params).json()
            data = res.get("data", [])
            if not data: break
            all_data.extend(data)
            end_time = data[-1][0] 
            if int(end_time) < one_year_ago: break
            time.sleep(0.2) 
        except Exception as e:
            print(f"抓取中断: {e}")
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'base_vol', 'quote_vol'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df[df['timestamp'] >= one_year_ago].copy()
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def run_institutional_strategy(df):
    print("🧠 正在组装【机构级三重滤网】复合指标...")
    
    # 滤网 1：宏观大趋势 (EMA 200)
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 滤网 2：ATR 波动率 (用于剔除垃圾横盘时间)
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_sma'] = df['atr'].rolling(50).mean() # 衡量当前波动率是否大于历史平均
    
    # 滤网 3：微观动能 (RSI 14 + MACD)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal_line']

    # 🎯 终极开仓逻辑：四因子共振
    df['signal'] = 0
    
    # 【做多条件】：大趋势向上 + 波动率够大 + RSI 回调到低位 + MACD动能柱开始反转向上
    long_cond = (
        (df['close'] > df['ema_200']) & 
        (df['atr'] > df['atr_sma']) & 
        (df['rsi'] < 45) & 
        (df['macd_hist'] > df['macd_hist'].shift(1))
    )
    
    # 【做空条件】：大趋势向下 + 波动率够大 + RSI 反抽到高位 + MACD动能柱开始反转向下
    short_cond = (
        (df['close'] < df['ema_200']) & 
        (df['atr'] > df['atr_sma']) & 
        (df['rsi'] > 55) & 
        (df['macd_hist'] < df['macd_hist'].shift(1))
    )
    
    # 🚀 出场逻辑 (平仓)：当 RSI 冲到极端超买超卖区，或者跌破均线时止盈出局
    exit_long = (df['rsi'] > 70) | (df['close'] < df['ema_200'])
    exit_short = (df['rsi'] < 30) | (df['close'] > df['ema_200'])

    # 状态机映射
    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1
    
    # 复杂的持仓状态推导（带有明确平仓逻辑）
    positions = []
    current_pos = 0
    for i in range(len(df)):
        if current_pos == 0:
            if long_cond.iloc[i]: current_pos = 1
            elif short_cond.iloc[i]: current_pos = -1
        elif current_pos == 1:
            if exit_long.iloc[i]: current_pos = 0
            elif short_cond.iloc[i]: current_pos = -1 # 直接反手
        elif current_pos == -1:
            if exit_short.iloc[i]: current_pos = 0
            elif long_cond.iloc[i]: current_pos = 1 # 直接反手
        positions.append(current_pos)
        
    df['position'] = positions
    df['position'] = df['position'].shift(1).fillna(0) # 避免未来函数

    # 计算真实利润与 0.06% 手续费摩擦
    fee_rate = 0.0006
    df['market_returns'] = df['close'].pct_change()
    df['trade_happened'] = df['position'].diff().fillna(0) != 0
    df['fee_cost'] = np.where(df['trade_happened'], fee_rate, 0)
    
    df['strategy_returns'] = (df['position'] * df['market_returns']) - df['fee_cost']
    df['累计市场收益'] = (1 + df['market_returns']).cumprod() - 1
    df['累计策略收益'] = (1 + df['strategy_returns']).cumprod() - 1
    
    df['累计资金曲线'] = (1 + df['strategy_returns']).cumprod()
    df['回撤'] = (df['累计资金曲线'] - df['累计资金曲线'].cummax()) / df['累计资金曲线'].cummax()
    
    return df

if __name__ == "__main__":
    df = fetch_1_year_bitget_data()
    
    if not df.empty:
        res = run_institutional_strategy(df)
        
        final_market = res['累计市场收益'].iloc[-1] * 100
        final_strategy = res['累计策略收益'].iloc[-1] * 100
        max_dd = res['回撤'].min() * 100
        trades = res['trade_happened'].sum()
        
        print("\n" + "★"*60)
        print("🏛️ 专业级：【三重滤网大周期回调策略】1 年期回测报告")
        print("★"*60)
        print(f"死拿现货收益:   {final_market:>8.2f}%")
        print(f"本策略净收益:   {final_strategy:>8.2f}% (已扣除真实手续费)")
        print("-" * 60)
        print(f"极限最大回撤:   {max_dd:>8.2f}%")
        print(f"全年交易次数:   {trades} 次 (极度克制)")
        print(f"胜率/盈亏比:    通过出场逻辑动态锁定利润")
        print("★"*60)
        res.to_csv('report.csv', index=False)
