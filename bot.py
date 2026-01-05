import time
import numpy as np
import pandas as pd
from binance.client import Client
from ta.trend import EMAIndicator
import ta

client = Client(API_KEY, API_SECRET)

last_trade_range = None
in_position = False

# ================= DATA =================

def get_klines(limit=300):
    klines = client.get_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=limit
    )
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume',
        '_','_','_','_','_','_'
    ])
    df[['open','high','low','close','volume']] = df[
        ['open','high','low','close','volume']
    ].astype(float)
    return df

# ================= TICK VOLUME PROFILE =================

def tick_volume_profile():
    trades = client.get_agg_trades(symbol=SYMBOL, limit=1000)

    prices = np.array([float(t['p']) for t in trades])
    volumes = np.array([float(t['q']) for t in trades])

    hist, bins = np.histogram(prices, bins=100, weights=volumes)
    centers = (bins[:-1] + bins[1:]) / 2

    profile = pd.DataFrame({
        'price': centers,
        'volume': hist
    }).sort_values('volume', ascending=False)

    total_vol = profile['volume'].sum()
    cum = 0
    value_prices = []

    for _, r in profile.iterrows():
        cum += r['volume']
        value_prices.append(r['price'])
        if cum >= total_vol * VALUE_AREA:
            break

    VAH = max(value_prices)
    VAL = min(value_prices)

    HVN = profile.iloc[0]['price']
    LVN = profile.iloc[-1]['price']

    return VAH, VAL, HVN, LVN

# ================= TREND FILTER =================

def trend_filter(df):
    ema_fast = EMAIndicator(df['close'], EMA_FAST).ema_indicator()
    ema_slow = EMAIndicator(df['close'], EMA_SLOW).ema_indicator()
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume']
    ).volume_weighted_average_price()

    price = df['close'].iloc[-1]

    if price > vwap.iloc[-1] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return "LONG"
    if price < vwap.iloc[-1] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return "SHORT"
    return None

# ================= RANGE CHECK =================

def is_range(df):
    high, low = df['high'].max(), df['low'].min()
    mid = (high + low) / 2
    return (high - low) / mid < 0.004

# ================= SIGNAL =================

def signal_logic(price, VAH, VAL, HVN, LVN, trend):
    buffer = (VAH - VAL) * 0.1

    if trend == "LONG" and price <= VAL + buffer and abs(price - LVN) < buffer:
        return "LONG"

    if trend == "SHORT" and price >= VAH - buffer and abs(price - LVN) < buffer:
        return "SHORT"

    return None

# ================= TRADE =================

def place_trade(side, price, VAH, VAL):
    global in_position
    in_position = True

    if side == "LONG":
        sl = VAL * (1 - SL_BUFFER)
        tp = VAH * (1 - TP_BUFFER)
        entry = Client.SIDE_BUY
        exit_side = Client.SIDE_SELL
    else:
        sl = VAH * (1 + SL_BUFFER)
        tp = VAL * (1 + TP_BUFFER)
        entry = Client.SIDE_SELL
        exit_side = Client.SIDE_BUY

    print(f"🚀 {side} | Entry {price:.2f} | SL {sl:.2f} | TP {tp:.2f}")

    client.create_order(
        symbol=SYMBOL,
        side=entry,
        type=Client.ORDER_TYPE_MARKET,
        quantity=POSITION_SIZE
    )

    client.create_order(
        symbol=SYMBOL,
        side=exit_side,
        type=Client.ORDER_TYPE_LIMIT,
        quantity=POSITION_SIZE,
        price=f"{tp:.2f}",
        timeInForce="GTC"
    )

    client.create_order(
        symbol=SYMBOL,
        side=exit_side,
        type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
        quantity=POSITION_SIZE,
        stopPrice=f"{sl:.2f}",
        price=f"{sl:.2f}",
        timeInForce="GTC"
    )

# ================= MAIN LOOP =================

while True:
    try:
        df = get_klines()
        price = df['close'].iloc[-1]

        if not is_range(df):
            in_position = False
            time.sleep(CHECK_INTERVAL)
            continue

        VAH, VAL, HVN, LVN = tick_volume_profile()
        trend = trend_filter(df)

        if not in_position:
            signal = signal_logic(price, VAH, VAL, HVN, LVN, trend)
            if signal:
                place_trade(signal, price, VAH, VAL)

        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("⚠️ ERROR:", e)
        time.sleep(10)
