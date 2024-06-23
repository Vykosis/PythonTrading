import pandas as pd
import backtrader as bt
from datetime import datetime
import matplotlib as plt

# Data loading and preprocessing
df = pd.read_csv("XAUUSD_H1.csv", parse_dates=True)
df["Date"] = pd.to_datetime(df["Date"])

# Custom data feed class to load data from pandas DataFrame
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
    )

def convert_to_heikin_ashi(df):
    """
    Convert a DataFrame with OHLC data to Heikin-Ashi candles.
    
    :param df: pandas DataFrame with 'Open', 'High', 'Low', 'Close' columns
    :return: pandas DataFrame with Heikin-Ashi OHLC data
    """
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    df['Close'] = ha_close
    
    # Calculate HA_Open
    df['Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    
    # For the first row, use the original open price
    df.loc[df.index[0], 'Open'] = df['Open'].iloc[0]

    # Calculate HA_High and HA_Low
    df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

    return df

# Custom indicator: Zero Lag Simple Moving Average
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
        
        # Calculate Zero Lag SMA
        zero_lag_sma = 2 * price - lag_price
        ema_multiplier = 2 / (self.p.period + 1)
        
        if len(self) == self.p.period:
            self.lines.zlsma[0] = zero_lag_sma
        else:
            prev_zlsma = self.lines.zlsma[-1]
            self.lines.zlsma[0] = (zero_lag_sma - prev_zlsma) * ema_multiplier + prev_zlsma

# Custom indicator: Chandelier Exit
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
        # Calculate long and short exit levels
        self.lines.long_exit[0] = self.highest_high[0] - self.p.atr_multiplier * self.atr[0]
        self.lines.long_exit[0] = max(self.lines.long_exit[0], self.lines.long_exit[-1]) if self.data.close[-1] > self.lines.long_exit[-1] else self.lines.long_exit[0]
        self.lines.short_exit[0] = self.lowest_low[0] + self.p.atr_multiplier * self.atr[0]
        self.lines.short_exit[0] = min(self.lines.short_exit[0], self.lines.short_exit[-1]) if self.data.close[-1] < self.lines.short_exit[-1] else self.lines.short_exit[0]

        prev_direction = self.lines.direction[-1]
                
        # Determine new direction
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

# Main trading strategy
class XAUUSDStrategy(bt.Strategy):
    params = (
        ('stake', 1),  # Number of units to trade
    )

    def __init__(self):
        self.chandelier_exit = ChandelierExit()
        self.zlsma = ZeroLagSMA()
        self.order = None
        self.buysell = None
        
        # For plotting
        self.buy_markers = []
        self.sell_markers = []

        # New variables for position management
        self.stop_loss = None
        self.take_profit = None
        self.initial_position_size = None

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
                self.buysell = 'buy'
                self.initial_position_size = self.params.stake
                self.order = self.buy(size=self.initial_position_size)
                self.buy_markers.append(len(self))
                self.set_stop_loss_take_profit('buy')
            elif sell_signal:
                self.buysell = 'sell'
                self.initial_position_size = self.params.stake
                self.order = self.sell(size=self.initial_position_size)
                self.sell_markers.append(len(self))
                self.set_stop_loss_take_profit('sell')
        
        else:
            # We are in the market, look for a signal to CLOSE trades
            if self.position.size > 0:  # Long position
                if sell_signal:  # Close long position on sell signal
                    self.buysell = 'close'
                    self.order = self.close()
                    self.sell_markers.append(len(self))
                else:
                    self.manage_position('buy')
            else:  # Short position
                if buy_signal:  # Close short position on buy signal
                    self.buysell = 'close'
                    self.order = self.close()
                    self.buy_markers.append(len(self))
                else:
                    self.manage_position('sell')

    def set_stop_loss_take_profit(self, position_type):
        # Calculate stop loss based on previous 10 candles
        if position_type == 'buy':
            self.stop_loss = min(self.data.low.get(ago=-1, size=10))
        else:  # sell
            self.stop_loss = max(self.data.high.get(ago=-1, size=10))

        # Calculate take profit as 1.5x the stop loss distance
        stop_loss_distance = abs(self.data.close[0] - self.stop_loss)
        self.take_profit = self.data.close[0] + (1.5 * stop_loss_distance) if position_type == 'buy' else self.data.close[0] - (1.5 * stop_loss_distance)

    def manage_position(self, position_type):
        # Check for stop loss
        if (position_type == 'buy' and self.data.low[0] <= self.stop_loss) or \
           (position_type == 'sell' and self.data.high[0] >= self.stop_loss):
            self.order = self.close()
            # self.log(f'Stop Loss hit. Closing position at {self.data.close[0]}')
        
        # Check for take profit (close 50% of position)
        elif (position_type == 'buy' and self.data.high[0] >= self.take_profit) or \
             (position_type == 'sell' and self.data.low[0] <= self.take_profit):
            if self.position.size == self.initial_position_size:  # Only if we haven't taken partial profits yet
                close_size = self.position.size * 25 // 100
                self.order = self.close(size=close_size)
                # self.log(f'Take Profit hit. Closing 50% of position at {self.data.close[0]}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                pass
                # self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                pass
                # self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
            # self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        # self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def stop(self):
        pass
        # self.log(f'Ending Value {self.broker.getvalue():.2f}')
            
def print_analyzer_results(analyzer):
    """
    Function to print the results of an analyzer in a formatted way.
    """
    print(f"--- {analyzer.__class__.__name__} ---")
    analysis = analyzer.get_analysis()
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    print()

def main():
    # Initialize Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add our strategy
    cerebro.addstrategy(XAUUSDStrategy)

    # Disable all default observers
    cerebro.stdstats = False
    
    # Set our desired cash start
    cerebro.broker.setcash(10000)

    # Create a Data Feed
    convert_to_heikin_ashi(df)
    data = PandasData(dataname=df)
    cerebro.adddata(data)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')

    # Run the strategy
    results = cerebro.run()
    cerebro.plot(style='candle', barup='green', bardown='red', volume=False, 
                    plotdist=0.1, subplot=False, iplot=False)

    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Print out all the statistics
    strat = results[0]
    for name, analyzer in strat.analyzers.getitems():
        print_analyzer_results(analyzer)

if __name__ == "__main__":
    main()