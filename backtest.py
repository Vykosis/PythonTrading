import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np

def chandelier_exit(df, atr_length=1, roll_length=1, mult=2):
    df.columns = df.columns.str.lower()
    my_atr = ta.Strategy(
        name="atr",
        ta=[{"kind": "atr", "length": atr_length, "col_names": ("atr",)}]
    )
    # Run it
    df.ta.strategy(my_atr, append=True)

    df['chandelier_long'] = df.rolling(roll_length)["ha_high"].max() - df.iloc[-1]["atr"] * mult
    df['chandelier_short'] = df.rolling(roll_length)["ha_low"].min() + df.iloc[-1]["atr"] * mult
    df.loc[df['ha_close'] > df['chandelier_long'].shift(1), 'chd_dir'] = 1
    df.loc[df['ha_close'] < df['chandelier_short'].shift(1), 'chd_dir'] = -1
    # chd = df[['chandelier_long', 'chandelier_short', 'chd_dir']]
    return df

df = pd.read_csv("XAUUSD_M5.csv", index_col="Date")

df.ta.ha(append=True)
zlma = ta.zlma(df.HA_close, 50)
df = chandelier_exit(df)
currentLong = False
currentShort = False
buy_markers = pd.DataFrame({"date":[],
                            "price":[]})

fig = go.Figure(go.Candlestick(
    x=df.index,
    open=df.ha_open,
    high=df.ha_high,
    low=df.ha_low,
    close=df.ha_close,
    name="Candles"
))

# Check for buy or sell
for index, row in df.iterrows():
    if (np.where(df.index==index)[0] == 0):
        continue
    current_index = np.where(df.index==index)[0]
    buy_cond1 = row.ha_close > zlma.loc[index]
    df2 = df.shift(1)
    buy_cond2 = row.chd_dir == 1 and df2.at[index,"chd_dir"] == -1
    if (buy_cond1 and buy_cond2):
        buy_markers.loc[len(buy_markers.index)] = [index,row.ha_close]
    
print(buy_markers)
fig.add_trace(go.Scatter(
    x = buy_markers.date,
    y = buy_markers.price,
    name = "Buy Markers",
    mode="markers"
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=zlma,
    name="ZLMA",
    line=dict(color="black")
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df.chandelier_long,
    name="Chandelier Long",
    line=dict(color="green")
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df.chandelier_short,
    name="Chandelier Short",
    line=dict(color="red")
))
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()