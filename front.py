import streamlit as st
import pandas as pd
import numpy as np

import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Add title and description
st.title("Stock Price Visualization and Prediction")
st.markdown("Select a stock symbol and date range to visualize historical data and see price predictions using an LSTM model.")

# Create sidebar for inputs
st.sidebar.header("Input Parameters")

# Stock symbol selection
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL")

# Date range selection
today = datetime.now().date()
one_year_ago = today - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", one_year_ago)
end_date = st.sidebar.date_input("End Date", today)

# Function to load and preprocess data
@st.cache_data
def load_data(symbol, start_date, end_date):
    try:
        # Download the data
        df = yf.download(symbol, start=start_date, end=end_date)
        df.columns = df.columns.droplevel('Ticker')
        df.reset_index(inplace=True)
        if len(df) == 0:
            st.error(f"No data found for {symbol}. Please check the stock symbol and try again.")
            return None

        # Add some technical indicators
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Volume moving average
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
        return None

# Function to prepare data for LSTM
def prepare_lstm_data(df, look_back=60):
    # Extract only the Close price
    data = df['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for prediction (last look_back days)
    x_test = []
    if len(scaled_data) > look_back:
        x_test.append(scaled_data[-look_back:, 0])
    else:
        # Handle case when not enough data is available
        padding = np.zeros(look_back - len(scaled_data))
        x_test.append(np.concatenate((padding, scaled_data[:, 0])))

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test, scaler, data

# Function to make predictions
def predict_future(model, x_test, scaler, forecast_days=30):
    # Make a copy of the last sequence
    curr_seq = x_test[0].copy()
    future_predictions = []

    # Predict next 'forecast_days' values
    for _ in range(forecast_days):
        # Get prediction for the next day (scaled)
        pred = model.predict(curr_seq.reshape(1, curr_seq.shape[0], 1), verbose=0)

        # Add the prediction to our list
        future_predictions.append(pred[0, 0])

        # Update the sequence by removing the first element and adding the new prediction
        curr_seq = np.append(curr_seq[1:], pred[0, 0])

    # Rescale predictions back to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions

# Function to load model with appropriate error handling
def recreate_lstm_model():
    """Recreate the LSTM model with the same architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    st.warning("Using a recreated LSTM model (not your pre-trained model)")
    return model

def load_lstm_model():
    try:
        model = recreate_lstm_model()
        return model
    except Exception as e:
        st.error(f"Error recreating model: {e}")
        return None

# Main execution
if st.sidebar.button("Load Data and Predict"):
    with st.spinner('Loading data and generating predictions...'):
        # Load stock data
        df = load_data(symbol, start_date, end_date)

        if df is not None:
            try:
                # Create two columns for the layout
                col1, col2 = st.columns(2)

                # Display basic info in the first column
                with col1:
                    st.subheader(f"Stock Information: {symbol}")
                    st.write("Data Range:", start_date, "to", end_date)
                    st.write("Latest Close Price: $", round(df['Close'].iloc[-1], 2))

                    # Calculate some stats
                    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                    price_change_percent = (price_change / df['Close'].iloc[0]) * 100

                    if price_change >= 0:
                        st.write("Overall Change: ",
                                f"📈 +${round(price_change, 2)} (+{round(price_change_percent, 2)}%)")
                    else:
                        st.write("Overall Change: ",
                                f"📉 -${round(abs(price_change), 2)} ({round(price_change_percent, 2)}%)")

                # Show recent data in the second column
                with col2:
                    st.subheader("Recent Stock Data")
                    st.dataframe(df.tail().style.format({"Open": "${:.2f}",
                                                         "High": "${:.2f}",
                                                         "Low": "${:.2f}",
                                                         "Close": "${:.2f}",
                                                         "Adj Close": "${:.2f}",
                                                         "MA5": "${:.2f}",
                                                         "MA20": "${:.2f}",
                                                         "MA50": "${:.2f}"}))

                # Create interactive plot with Plotly
                st.subheader("Historical Price and Volume")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.1,
                                    subplot_titles=('Price', 'Volume'),
                                    row_heights=[0.7, 0.3])

                # Add price candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="OHLC"
                    ),
                    row=1, col=1
                )

                # Add moving averages
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA5'],
                        name="MA5",
                        line=dict(color='purple', width=1)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA20'],
                        name="MA20",
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA50'],
                        name="MA50",
                        line=dict(color='green', width=1)
                    ),
                    row=1, col=1
                )

                # Add volume bars - Fixed coloring logic
                colors = []
                for i in range(len(df)):
                    if i > 0:
                        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                            colors.append('green')
                        else:
                            colors.append('red')
                    else:
                        colors.append('green')

                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        marker_color=colors,
                        name="Volume"
                    ),
                    row=2, col=1
                )

                # Add volume moving average
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Volume_MA10'],
                        name="Volume MA10",
                        line=dict(color='black', width=1)
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    title=f"{symbol} Stock Price and Volume",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Load the LSTM model
                st.info("Loading LSTM model...")
                model = load_lstm_model()

                if model is not None:
                    # Prepare data for prediction
                    x_test, scaler, original_data = prepare_lstm_data(df)
                    
                    # Make future predictions
                    forecast_days = 30  # Predict for the next 30 days
                    future_pred = predict_future(model, x_test, scaler, forecast_days)

                    # Create date range for the prediction
                    last_date = df.index[-1]
                    if not isinstance(last_date, (pd.Timestamp, datetime)):
    # If it's not a datetime, convert it to one
                        try:
                            last_date = pd.to_datetime(last_date)
                        except:
        # If conversion fails, use today's date
                            last_date = pd.Timestamp.today()
                            st.warning("Could not determine the last date from data, using today's date instead.")

#                            Now create the prediction dates
                    prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

                    # Create a dataframe for predictions
                    forecast_df = pd.DataFrame({
                        'Predicted_Close': future_pred.flatten()
                    }, index=prediction_dates)

                    # Plot the historical and predicted prices
                    st.subheader(f"LSTM Model Prediction for {symbol} (Next {forecast_days} Days)")

                    fig = go.Figure()

                    # Add historical data
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Historical Close Price',
                            line=dict(color='blue')
                        )
                    )

                    # Add predicted data
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Predicted_Close'],
                            mode='lines',
                            name='Predicted Close Price',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    # Add a vertical line to separate historical from prediction
                    fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="black")

                    # Update layout
                    fig.update_layout(
                        title=f"{symbol} Stock Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display prediction data
                    st.subheader("Prediction Data")
                    st.dataframe(forecast_df.style.format({"Predicted_Close": "${:.2f}"}))

                    # Calculate predicted returns
                    current_price = df['Close'].iloc[-1]
                    future_price = forecast_df['Predicted_Close'].iloc[-1]
                    predicted_change = future_price - current_price
                    predicted_percent = (predicted_change / current_price) * 100

                    st.subheader("Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric(f"Predicted Price ({forecast_days} days)", f"${future_price:.2f}")
                    with col3:
                        st.metric("Predicted Change",
                                f"{predicted_change:.2f} ({predicted_percent:.2f}%)",
                                delta=f"{predicted_percent:.2f}%")
                else:
                    # Model could not be loaded, offer fallback option
                    st.error("Could not load the LSTM model.")

                    # Show alternative option for a demo
                    st.warning("Would you like to see a demo with a simple random prediction model instead?")
                    if st.button("Use Demo Prediction"):
                        # Create a simple prediction function for demonstration
                        def demo_predict(last_price, forecast_days=30):
                            predictions = []
                            current = last_price
                            for _ in range(forecast_days):
                                # Random walk with slight upward bias
                                change = np.random.normal(0.001, 0.02) * current
                                current += change
                                predictions.append(current)
                            return np.array(predictions).reshape(-1, 1)

                        # Get the last price
                        last_price = df['Close'].iloc[-1]

                        # Generate demo predictions
                        future_pred = demo_predict(last_price, forecast_days=30)

                        # Rest of the prediction display code (same as above)
                        # Create date range for the prediction
                        last_date = df.index[-1]
                        prediction_dates = pd.date_range(start=last_date + timedelta(days=1),
                                                        periods=30)

                        # Create a dataframe for predictions
                        forecast_df = pd.DataFrame({
                            'Predicted_Close': future_pred.flatten()
                        }, index=prediction_dates)

                        # Display the demo prediction chart and data
                        st.subheader("DEMO Prediction (Not LSTM model)")

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['Close'],
                                mode='lines',
                                name='Historical Close Price',
                                line=dict(color='blue')
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['Predicted_Close'],
                                mode='lines',
                                name='Demo Prediction (Random Walk)',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="black")
                        fig.update_layout(
                            title=f"{symbol} Stock Price - Demo Prediction",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        st.warning("This is a DEMO prediction using a random walk model, not the LSTM model.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app visualizes stock data and provides predictions using a pre-trained LSTM model.
- Data is fetched from Yahoo Finance using the yfinance library
- Predictions are made using a deep learning LSTM model
- The model must be saved as 'lstm_model.h5' in the same directory as this app
""")

# Add a warning about predictions
st.sidebar.warning("""
⚠️ Disclaimer: Stock price predictions are for educational purposes only and should not be used for financial decisions.
""")

# Add instructions for model creation
with st.sidebar.expander("Need to create a model?"):
    st.markdown("""
    If you don't have an LSTM model, you need to:

    1. Train an LSTM model on historical stock data
    2. Save the model using `model.save('lstm_model.h5')`
    3. Place the model file in the same directory as this script

    You can find tutorials on creating LSTM models for stock prediction online.
    """)
