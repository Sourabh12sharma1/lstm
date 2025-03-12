import pandas as pd
import numpy as np
import yfinance as yf

def calculate_supertrend_v2(df, super_trend_lookback=30, super_trend_multiplier=10, span_12=12, span_26=26, sma_window=20):
    """
    Calculate SuperTrend-V2 indicator along with additional signals.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    super_trend_lookback : int
        Lookback period for ATR calculation
    super_trend_multiplier : float
        Multiplier for ATR in SuperTrend calculation
    span_12 : int
        Span for 12-day EMA
    span_26 : int
        Span for 26-day EMA
    sma_window : int
        Window for Simple Moving Average

    Returns:
    --------
    pandas.DataFrame
        DataFrame with SuperTrend-V2 indicator and signals
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Calculate EMAs
    df['EMA-12'] = df['Close'].ewm(span=span_12, min_periods=span_12, adjust=True).mean()
    df['EMA-26'] = df['Close'].ewm(span=span_26, min_periods=span_26, adjust=True).mean()

    # Calculate SMA
    df['ST_SMA'] = df['Close'].rolling(window=sma_window, min_periods=0).mean()

    # Calculate Signal and Position
    df['ST_SignalRes'] = np.where(df['ST_SMA'] > df['Close'], 0, 1)
    df['ST_Position'] = df['ST_SignalRes'].diff()

    # Calculate ATR
    price_diffs = [df['High'] - df['Low'],
                   df['High'] - df['Close'].shift(),
                   df['Low'] - df['Close'].shift()]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    atr = true_range.ewm(alpha=1/super_trend_lookback).mean()

    # Calculate Basic Upper and Lower Bands
    hl2 = (df['High'] + df['Low']) / 2
    basic_upper_band = hl2 + super_trend_multiplier * atr
    basic_lower_band = hl2 - super_trend_multiplier * atr

    # Initialize Final Bands and SuperTrend
    final_upper_band = [0] * len(df)
    final_lower_band = [0] * len(df)
    super_trend = [0] * len(df)

    # Calculate Final Bands and SuperTrend
    for idx in range(1, len(df)):
        prev, curr = idx - 1, idx

        # Final Upper Band
        if (basic_upper_band[curr] < final_upper_band[prev]) or (df['Close'][prev] > final_upper_band[prev]):
            final_upper_band[curr] = basic_upper_band[curr]
        else:
            final_upper_band[curr] = final_upper_band[prev]

        # Final Lower Band
        if (basic_lower_band[curr] > final_lower_band[prev]) or (df['Close'][prev] < final_lower_band[prev]):
            final_lower_band[curr] = basic_lower_band[curr]
        else:
            final_lower_band[curr] = final_lower_band[prev]

        # SuperTrend-V2
        if (super_trend[prev] == final_upper_band[prev]) and (df['Close'][curr] < final_upper_band[curr]):
            super_trend[curr] = final_upper_band[curr]
        elif (super_trend[prev] == final_upper_band[prev]) and (df['Close'][curr] > final_upper_band[curr]):
            super_trend[curr] = final_lower_band[curr]
        elif (super_trend[prev] == final_lower_band[prev]) and (df['Close'][curr] > final_lower_band[curr]):
            super_trend[curr] = final_lower_band[curr]
        elif (super_trend[prev] == final_lower_band[prev]) and (df['Close'][curr] < final_lower_band[curr]):
            super_trend[curr] = final_upper_band[curr]

    df['SuperTrend-V2'] = super_trend
    df['final_upper_band'] = pd.Series(final_upper_band)
    df['final_lower_band'] = pd.Series(final_lower_band)

    # Calculate Buy and Sell Signals
    action_val = {'BUY': 1, 'SELL': -1}
    super_trend_v2_signals = ['SuperTrend-V2-BUY', 'SuperTrend-V2-SELL']

    for signal in super_trend_v2_signals:
        _, _, action = signal.split('-')
        if action == 'BUY':
            df[signal] = np.where(df['Close'] > df['final_upper_band'], action_val[action], 0)
        elif action == 'SELL':
            df[signal] = np.where(df['Close'] < df['final_lower_band'], action_val[action], 0)

    # Calculate Combined Buy/Sell Signal
    df['buy_sell'] = df['SuperTrend-V2-BUY'] + df['SuperTrend-V2-SELL']
    df['buy_sell_fill'] = df['buy_sell'].replace(0, method='ffill')

    # Finalize Buy and Sell Signals
    df['SuperTrend-V2-BUY'] = df['buy_sell_fill'].apply(lambda x: x if x == 1 else 0)
    df['SuperTrend-V2-SELL'] = df['buy_sell_fill'].apply(lambda x: x if x == -1 else 0)

    # Clean up temporary columns
    
    return df

# Example usage for AAPL
def main():
    # Download AAPL data
    ticker = "CHTR"
    period = "10y"
    
    print(f"Downloading {ticker} data for {period}...")
    data = yf.download(ticker, period=period)
    data.columns = data.columns.droplevel('Ticker')
    data.reset_index(inplace=True)
    # Calculate SuperTrend-V2
    print("Calculating SuperTrend-V2...")
    result = calculate_supertrend_v2(data)
    result.to_csv(f"{ticker}_supertrend_v2_neww.csv")
    # Display the first few rows of the result
    print("\nFirst few rows of the result:")
    print(result[['Close', 'SuperTrend-V2', 'SuperTrend-V2-BUY', 'SuperTrend-V2-SELL']].head(10))
    
    # Display the last few rows of the result
    print("\nLast few rows of the result:")
    print(result[['Close', 'SuperTrend-V2', 'SuperTrend-V2-BUY', 'SuperTrend-V2-SELL']].tail(10))

if __name__ == "__main__":
    main()