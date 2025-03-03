import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Add title and description
st.title("Stock Price Visualization and Prediction")
st.markdown("Select a stock symbol and date range to visualize historical data and make price predictions using an LSTM model.")

# Create sidebar for inputs
st.sidebar.header("Input Parameters")

# Stock symbol selection
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", "GOOGL")

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
        df['SMA7'] = df['Close'].rolling(window=7).mean()
        df['SMA21'] = df['Close'].rolling(window=21).mean()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # Volume moving average
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()

        # Drop NaN values
        df.dropna(inplace=True)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
        return None

# Function to prepare data for LSTM
def prepare_lstm_data(df, sequence_length=30):
    features = ['Close', 'SMA7', 'SMA21']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y_trend, y_price = [], [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y_trend.append(df['Trend'].iloc[i])
        y_price.append(scaled_data[i, 0])  # Close price

    X, y_trend, y_price = np.array(X), np.array(y_trend), np.array(y_price)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_trend_train, y_trend_test = y_trend[:split], y_trend[split:]
    y_price_train, y_price_test = y_price[:split], y_price[split:]

    return X_train, X_test, y_trend_train, y_trend_test, y_price_train, y_price_test, scaler, scaled_data

# Function to build LSTM model
def build_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = LSTM(32)(x)
    x = Dense(16, activation='relu')(x)

    # Two outputs
    trend_output = Dense(1, activation='sigmoid', name='trend_output')(x)
    price_output = Dense(1, activation='linear', name='price_output')(x)

    model = Model(inputs=input_layer, outputs=[trend_output, price_output])

    model.compile(loss={'trend_output':'binary_crossentropy', 'price_output':'mse'},
                  optimizer='adam',
                  metrics={'trend_output':'accuracy', 'price_output':'mse'})

    return model

# Function to train LSTM model
def train_lstm_model(X_train, y_trend_train, y_price_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, {'trend_output': y_trend_train, 'price_output': y_price_train},
                        epochs=80, batch_size=64, validation_split=0.1, verbose=0)
    return model, history

# Function to make predictions
def predict_future(model, last_sequence, scaler, forecast_days=15):
    predictions_trend = []
    predictions_price = []

    current_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

    for _ in range(forecast_days):
        pred_trend, pred_price = model.predict(current_sequence, verbose=0)
        predictions_trend.append(int(pred_trend[0][0] > 0.5))
        predictions_price.append(pred_price[0][0])

        new_close = pred_price[0][0]
        new_row = np.array([new_close, 0, 0])
        current_sequence = np.append(current_sequence[:, 1:, :], [[new_row]], axis=1)

    predicted_prices = scaler.inverse_transform(
        np.column_stack((predictions_price, np.zeros((forecast_days, 2))))
    )[:, 0]

    return predictions_trend, predicted_prices

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
                                                         "SMA7": "${:.2f}",
                                                         "SMA21": "${:.2f}",
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
                        y=df['SMA7'],
                        name="SMA7",
                        line=dict(color='purple', width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA21'],
                        name="SMA21",
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA5'],
                        name="MA5",
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA20'],
                        name="MA20",
                        line=dict(color='green', width=1)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MA50'],
                        name="MA50",
                        line=dict(color='red', width=1)
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

                # Create target variable for crossover (1 if SMA7 > SMA21 else 0)
                df['Trend'] = np.where(df['SMA7'] > df['SMA21'], 1, 0)

                # Prepare data for LSTM
                X_train, X_test, y_trend_train, y_trend_test, y_price_train, y_price_test, scaler, scaled_data = prepare_lstm_data(df)

                # Train the model
                st.info("Training LSTM model...")
                model, history = train_lstm_model(X_train, y_trend_train, y_price_train)
                st.success("Model training completed.")

                # Make future predictions
                forecast_days = 15  # Predict for the next 7 days
                last_sequence = scaled_data[-30:]
                predictions_trend, predicted_prices = predict_future(model, last_sequence, scaler, forecast_days)

                # Create date range for the prediction
                last_date1 = df["Date"].tolist()
                last_date = last_date1[-1]
                print("last#############4444444444444##############",last_date)
                if not isinstance(last_date, (pd.Timestamp, datetime)):
                    try:
                        last_date = pd.to_datetime(last_date)
                    except:
                        last_date = pd.Timestamp.today()
                        st.warning("Could not determine the last date from data, using today's date instead.")

                prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

                # Create a dataframe for predictions
                forecast_df = pd.DataFrame({
                    'Predicted_Close': predicted_prices,
                    'Predicted_Trend': ['Uptrend' if t == 1 else 'Downtrend' for t in predictions_trend]
                }, index=prediction_dates)

                # Plot the historical and predicted prices
                st.subheader(f"LSTM Model Prediction for {symbol} (Next {forecast_days} Days)")

                fig = go.Figure()
                print(df)
                print("forecast4444444444444444444444444#####################################################",forecast_df)
                # Add historical data
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
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
                        line=dict(color='red')
                    )
                )
               
                print("last###########################",last_date)
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

                # Display the prediction graph
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("LSTM Model Training History")

                # Create a figure with two y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add loss history
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['loss']) + 1)),
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['val_loss']) + 1)),
                        y=history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ),
                    secondary_y=False
                )

                # Add accuracy history
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['trend_output_accuracy']) + 1)),
                        y=history.history['trend_output_accuracy'],
                        mode='lines',
                        name='Training Accuracy',
                        line=dict(color='green')
                    ),
                    secondary_y=True
                )

                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history['val_trend_output_accuracy']) + 1)),
                        y=history.history['val_trend_output_accuracy'],
                        mode='lines',
                        name='Validation Accuracy',
                        line=dict(color='orange')
                    ),
                    secondary_y=True
                )

                # Update layout
                fig.update_layout(
                    title="LSTM Model Training History",
                    xaxis_title="Epoch",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Set y-axes titles
                fig.update_yaxes(title_text="Loss", secondary_y=False)
                fig.update_yaxes(title_text="Accuracy", secondary_y=True)
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

                # Display predicted trend
                st.subheader("Predicted Trend")
                for i, (date, trend) in enumerate(forecast_df['Predicted_Trend'].items(), 1):
                    st.write(f"Day {i} ({date.date()}): {trend}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app visualizes stock data and provides predictions using an LSTM model.
- Data is fetched from Yahoo Finance using the yfinance library
- Predictions are made using a deep learning LSTM model
- The model is trained on-the-fly and not saved
""")

# Add a warning about predictions
st.sidebar.warning("""
⚠️ Disclaimer: Stock price predictions are for educational purposes only and should not be used for financial decisions.
""")