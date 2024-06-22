import pandas as pd
import backtrader as bt
from datetime import datetime

# Data loading and preprocessing
df = pd.read_csv("XAUUSD_H1.csv", parse_dates=True)
df["Date"] = pd.to_datetime(df["Date"])

class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
    )

class ZeroLagSMA(bt.Indicator):
    lines = ('zlsma',)
    params = (('period', 50),)

    plotinfo = dict(
        plot=False,
        plotname='Zero Lag SMA',
        subplot=False,
        plotlinelabels=True
    )

    plotlines = dict(
        zlsma=dict(color='red', linewidth=2)
    )

    def __init__(self):
        self.addminperiod(self.p.period)
        self.lag = (self.p.period - 1) // 2
        self.sma = bt.indicators.SMA(self.data, period=self.p.period)

    def next(self):
        price = self.data[0]
        sma = self.sma[0]
        lag_price = self.data[-self.lag]
        
        zero_lag_sma = 2 * price - lag_price
        ema_multiplier = 2 / (self.p.period + 1)
        
        if len(self) == self.p.period:
            self.lines.zlsma[0] = zero_lag_sma
        else:
            prev_zlsma = self.lines.zlsma[-1]
            self.lines.zlsma[0] = (zero_lag_sma - prev_zlsma) * ema_multiplier + prev_zlsma

class ChandelierExit(bt.Indicator):
    lines = ('buy_marker', 'sell_marker', 'direction', 'long_exit', 'short_exit')
    params = (('period', 1), ('atr_multiplier', 2))

    plotinfo = dict(
        plot=False,
        plotname='Chandelier Exit',
        subplot=False,
        plotlinelabels=True
    )

    plotlines = dict(
        long_exit=dict(color='green', ls='--', linewidth=1),
        short_exit=dict(color='red', ls='--', linewidth=1),
        direction=dict(_plotskip=True),
        buy_marker=dict(_plotskip=True),
        sell_marker=dict(_plotskip=True)
    )

    def __init__(self):
        super(ChandelierExit, self).__init__()
        self.atr = bt.indicators.ATR(self.data, period=self.p.period)
        self.highest_high = bt.indicators.Highest(self.data.close, period=self.p.period)
        self.lowest_low = bt.indicators.Lowest(self.data.close, period=self.p.period)

    def next(self):
        self.lines.long_exit[0] = self.highest_high[0] - self.p.atr_multiplier * self.atr[0]
        self.lines.long_exit[0] = max(self.lines.long_exit[0], self.lines.long_exit[-1]) if self.data.close[-1] > self.lines.long_exit[-1] else self.lines.long_exit[0]
        self.lines.short_exit[0] = self.lowest_low[0] + self.p.atr_multiplier * self.atr[0]
        self.lines.short_exit[0] = min(self.lines.short_exit[0], self.lines.short_exit[-1]) if self.data.close[-1] < self.lines.short_exit[-1] else self.lines.short_exit[0]

        prev_direction = self.lines.direction[-1]
                
        if self.data.close[0] > self.lines.short_exit[-1]:
            new_direction = 1  # Bullish
        elif self.data.close[0] < self.lines.long_exit[-1]:
            new_direction = -1  # Bearish
        else:
            new_direction = prev_direction  # No change

        # Check for direction change and plot markers
        if new_direction == 1 and prev_direction <= 0:
            self.lines.buy_marker[0] = self.data.low[0]  # Plot buy marker at the low of the bar
        elif new_direction == -1 and prev_direction >= 0:
            self.lines.sell_marker[0] = self.data.high[0]  # Plot sell marker at the high of the bar

        self.lines.direction[0] = new_direction

class XAUUSDStrategy(bt.Strategy):
    params = (
        ('stake', 1),  # Number of units to trade
    )

    plotinfo = dict(subplot=False)

    def __init__(self):
        self.chandelier_exit = ChandelierExit()
        self.zlsma = ZeroLagSMA()
        self.order = None
        self.buysell = None
        
        # For plotting
        self.buy_markers = []
        self.sell_markers = []

    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Calculate signals
        buy_signal = (self.chandelier_exit.direction[0] == 1 and self.chandelier_exit.direction[-1] == -1) and (self.data.close > self.zlsma.zlsma[0])
        sell_signal = (self.chandelier_exit.direction[0] == -1 and self.chandelier_exit.direction[-1] == 1) and (self.data.close < self.zlsma.zlsma[0])

        # Check if we are in the market
        if not self.position:
            # We are not in the market, look for a signal to OPEN trades
            if buy_signal:
                self.log(f'BUY CREATE, {self.data.close[0]}')
                self.buysell = 'buy'
                self.order = self.buy(size=self.params.stake)
                self.buy_markers.append(len(self))  # Mark the buy point
            elif sell_signal:
                self.log(f'SELL CREATE, {self.data.close[0]}')
                self.buysell = 'sell'
                self.order = self.sell(size=self.params.stake)
                self.sell_markers.append(len(self))  # Mark the sell point
        
        else:
            # We are in the market, look for a signal to CLOSE trades
            if self.position.size > 0:  # Long position
                if sell_signal:  # Close long position on sell signal
                    self.log(f'CLOSE LONG, {self.data.close[0]}')
                    self.buysell = 'close'
                    self.order = self.close()
                    self.sell_markers.append(len(self))  # Mark the exit point
            else:  # Short position
                if buy_signal:  # Close short position on buy signal
                    self.log(f'CLOSE SHORT, {self.data.close[0]}')
                    self.buysell = 'close'
                    self.order = self.close()
                    self.buy_markers.append(len(self))  # Mark the exit point

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def stop(self):
        self.log(f'Ending Value {self.broker.getvalue():.2f}')

    def plot(self, plotter):
        for marker in self.buy_markers:
            plotter.plot(self.data.datetime.date(marker), 
                         self.data.low[marker], 
                         '^', markersize=10, color='g', fillstyle='full')
        for marker in self.sell_markers:
            plotter.plot(self.data.datetime.date(marker), 
                         self.data.high[marker], 
                         'v', markersize=10, color='r', fillstyle='full')

def main():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(XAUUSDStrategy)
    cerebro.broker.setcash(1000)

    data = PandasData(dataname=df)
    cerebro.adddata(data)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    cerebro.plot(style="candle", volume=False, barup='green', bardown='red')

if __name__ == "__main__":
    main()