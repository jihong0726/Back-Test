import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

VERSION = "V8.3_STABLE"

CONFIG = {
    "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT"],
    "interval": "5m",
    "bars": 3000,
    "fee": 0.0006,
    "slippage": 0.0002
}

REPORT_DIR = "reports"


def ensure_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


# -------------------------------------------------
# 数据抓取
# -------------------------------------------------

def fetch_bitget(symbol, limit=3000):

    url = "https://api.bitget.com/api/v2/mix/market/candles"

    params = {
        "symbol":symbol,
        "productType":"USDT-FUTURES",
        "granularity":"5m",
        "limit":"1000"
    }

    rows=[]
    end_time=int(time.time()*1000)

    while len(rows)<limit:

        params["endTime"]=str(end_time)

        r=requests.get(url,params=params,timeout=20)
        data=r.json().get("data",[])

        if not data:
            break

        rows+=data

        oldest=min(int(x[0]) for x in data)
        end_time=oldest-1

        time.sleep(0.1)

    df=pd.DataFrame(rows,columns=[
        "timestamp","open","high","low","close","volume","q"
    ])

    for c in ["open","high","low","close","volume"]:
        df[c]=pd.to_numeric(df[c])

    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="ms")

    df=df.sort_values("timestamp").reset_index(drop=True)

    return df.tail(limit)


# -------------------------------------------------
# 指标
# -------------------------------------------------

def calc_rsi(series,period=14):

    delta=series.diff()

    gain=(delta.where(delta>0,0)).rolling(period).mean()
    loss=(-delta.where(delta<0,0)).rolling(period).mean()

    rs=gain/loss

    rsi=100-(100/(1+rs))

    return rsi.fillna(50)


def calc_atr(df,period=14):

    high=df["high"]
    low=df["low"]
    close=df["close"]

    tr=np.maximum(
        high-low,
        np.maximum(
            abs(high-close.shift()),
            abs(low-close.shift())
        )
    )

    atr=tr.rolling(period).mean()

    return atr


def calc_vwap(df):

    price=(df["high"]+df["low"]+df["close"])/3
    pv=price*df["volume"]

    cum_pv=pv.cumsum()
    cum_vol=df["volume"].replace(0,np.nan).cumsum()

    vwap=cum_pv/cum_vol

    vwap=vwap.replace([np.inf,-np.inf],np.nan).ffill()

    return vwap


def add_indicators(df):

    df=df.copy()

    df["ema20"]=df["close"].ewm(span=20).mean()
    df["ema50"]=df["close"].ewm(span=50).mean()

    df["rsi"]=calc_rsi(df["close"],7)

    df["atr"]=calc_atr(df)

    df["vwap"]=calc_vwap(df)

    df["dist_ema20"]=df["close"]/df["ema20"]-1

    df["dist_vwap"]=df["close"]/df["vwap"]-1

    return df


# -------------------------------------------------
# 策略
# -------------------------------------------------

def generate_strategies(df):

    s={}

    s["RSI_20_80"]=np.where(df["rsi"]<20,1,
                   np.where(df["rsi"]>80,-1,0))

    s["RSI_25_75"]=np.where(df["rsi"]<25,1,
                   np.where(df["rsi"]>75,-1,0))

    s["EMA20_dev_1.5"]=np.where(df["dist_ema20"]<-0.015,1,
                        np.where(df["dist_ema20"]>0.015,-1,0))

    s["EMA20_dev_2"]=np.where(df["dist_ema20"]<-0.02,1,
                      np.where(df["dist_ema20"]>0.02,-1,0))

    s["VWAP_1.5"]=np.where(df["dist_vwap"]<-0.015,1,
                    np.where(df["dist_vwap"]>0.015,-1,0))

    return s


# -------------------------------------------------
# 回测
# -------------------------------------------------

def backtest(df,signal):

    position=0
    entry=0

    returns=[]

    for i in range(1,len(df)):

        price=df["close"].iloc[i]
        prev=df["close"].iloc[i-1]

        if position!=0:
            ret=position*((price/prev)-1)
        else:
            ret=0

        if signal[i-1]!=position:

            position=signal[i-1]

            if position!=0:
                entry=price

        returns.append(ret)

    r=pd.Series(returns)

    total=(1+r).prod()-1

    dd=((1+r).cumprod()/((1+r).cumprod().cummax())-1).min()

    sharpe=r.mean()/r.std() if r.std()!=0 else 0

    return total*100,dd*100,sharpe


# -------------------------------------------------
# 主程序
# -------------------------------------------------

def main():

    ensure_dir()

    summary=[]

    for sym in CONFIG["symbols"]:

        print("processing",sym)

        df=fetch_bitget(sym,CONFIG["bars"])

        df=add_indicators(df)

        strategies=generate_strategies(df)

        for name,signal in strategies.items():

            total,dd,sharpe=backtest(df,signal)

            summary.append({
                "symbol":sym,
                "strategy":name,
                "return":total,
                "dd":dd,
                "sharpe":sharpe
            })

    res=pd.DataFrame(summary)

    res=res.sort_values("return",ascending=False)

    now=datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path=f"{REPORT_DIR}/result_{now}.csv"

    res.to_csv(csv_path,index=False)

    with open("latest_summary.txt","w") as f:
        f.write(res.head(20).to_string())

    print(res.head(20))

    print("saved:",csv_path)


if __name__=="__main__":
    main()
