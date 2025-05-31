# Module responsible for building and training LSTM models for each commodity.

# Model Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from pathlib import Path
import warnings

# Agents Imports
import re
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# App imports
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set the seed to ensure reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

warnings.filterwarnings('ignore')

# Loading the datasets
print("=" * 50)
print("🧹 Preparing data... Please wait.")
print("=" * 50)

soybean_data = pd.read_csv('/commodities_datasets/soybean_data.csv', index_col=0, parse_dates=True)
corn_data = pd.read_csv('/commodities_datasets/corn_data.csv', index_col=0, parse_dates=True)
wheat_data = pd.read_csv('/commodities_datasets/wheat_data.csv', index_col=0, parse_dates=True)

# Yesterday Close (Will be used later)
yesterday_soy_price = soybean_data['Close'].iloc[-1]
yesterday_corn_price = corn_data['Close'].iloc[-1]
yesterday_wheat_price = wheat_data['Close'].iloc[-1]

# Time series (we will only use the "Close" column to make the prdictions)
soybean_serie = soybean_data.Close.values.reshape(-1, 1)
corn_serie = corn_data.Close.values.reshape(-1, 1)
wheat_serie = wheat_data.Close.values.reshape(-1, 1)

# Function to create the dataset for the LSTM model with look_back
# The look_back parameter defines the number of time steps the model will use to make each prediction
def create_dataset(data, look_back):
    # Initialize two empty lists, X and Y, which will contain the input and output sequences, respectively
    X, Y = [], []
    # Iterates through the dataset minus the look_back value. This is done to avoid out-of-bounds indexes of the array
    for i in range(len(data) - look_back):
        # Collect a sequence of data of length look_back starting at index i
        a = data[i:(i + look_back), 0]
        # Add the sequence to the list X
        X.append(a)
        # Add the value immediately after the look_back sequence to the list Y. This will be our output value (target).
        Y.append(data[i + look_back, 0])
    # Convert X and Y to numpy arrays for compatibility with most machine learning libraries
    return np.array(X), np.array(Y)

# Decorator that allows caching of resources in Streamlit to improve application performance
@st.cache_resource
def training_module():

    train_size = 0.95

    # Train test split
    soy_index = int(len(soybean_serie) * train_size)
    train_soy, test_soy = soybean_serie[0:soy_index, :], soybean_serie[soy_index:len(soybean_serie), :]
    corn_index = int(len(corn_serie) * train_size)
    train_corn, test_corn = corn_serie[0:corn_index, :], corn_serie[corn_index:len(corn_serie), :]
    wheat_index = int(len(wheat_serie) * train_size)
    train_wheat, test_wheat = wheat_serie[0:wheat_index, :], wheat_serie[wheat_index:len(wheat_serie), :]

    # Scalers
    soy_scaler = MinMaxScaler(feature_range=(0, 1))
    corn_scaler = MinMaxScaler(feature_range=(0, 1))
    wheat_scaler = MinMaxScaler(feature_range=(0, 1))

    # Train
    soy_train_sc = soy_scaler.fit_transform(train_soy)
    corn_train_sc = corn_scaler.fit_transform(train_corn)
    wheat_train_sc = wheat_scaler.fit_transform(train_wheat)

    # Test
    soy_test_sc = soy_scaler.transform(test_soy)
    corn_test_sc = corn_scaler.transform(test_corn)
    wheat_test_sc = wheat_scaler.transform(test_wheat)

    # Value referring to how many values in the past the model will look at to make the next prediction
    look_back = 10

    # Datasets for LSTM model
    X_soy_train, y_soy_train = create_dataset(soy_train_sc, look_back)
    X_soy_test, y_soy_test = create_dataset(soy_test_sc, look_back)
    X_corn_train, y_corn_train = create_dataset(corn_train_sc, look_back)
    X_corn_test, y_corn_test = create_dataset(corn_test_sc, look_back)
    X_wheat_train, y_wheat_train = create_dataset(wheat_train_sc, look_back)
    X_wheat_test, y_wheat_test = create_dataset(wheat_test_sc, look_back)

    # Reshape the data to [samples, time steps, features]. This is a requirement of the LSTM model
    X_soy_train = np.reshape(X_soy_train, (X_soy_train.shape[0], X_soy_train.shape[1], 1))
    X_soy_test = np.reshape(X_soy_test, (X_soy_test.shape[0], X_soy_train.shape[1], 1))
    X_corn_train = np.reshape(X_corn_train, (X_corn_train.shape[0], X_corn_train.shape[1], 1))
    X_corn_test = np.reshape(X_corn_test, (X_corn_test.shape[0], X_corn_train.shape[1], 1))
    X_wheat_train = np.reshape(X_wheat_train, (X_wheat_train.shape[0], X_wheat_train.shape[1], 1))
    X_wheat_test = np.reshape(X_wheat_test, (X_wheat_test.shape[0], X_wheat_train.shape[1], 1))

    # Building the models

    print("=" * 50)
    print("🧠 Building LSTM models... Please wait.")
    print("=" * 50)
    soy_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape = (look_back, 1)),
                                            tf.keras.layers.Dense(1)])

    corn_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape = (look_back, 1)),
                                             tf.keras.layers.Dense(1)])

    wheat_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape = (look_back, 1)),
                                              tf.keras.layers.Dense(1)])

    # Compile
    soy_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    corn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    wheat_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Callbacks (Useful when the model is trained for many epochs)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3, restore_best_weights=True)

    # Soybean fit
    print("=" * 50)
    print("🤖 Starting soybean model training... Please wait.")
    print("=" * 50)
    soy_model.fit(X_soy_train, y_soy_train, epochs = 10, batch_size = 1, verbose = 1, callbacks = [early_stopping])

    # Corn fit
    print("=" * 50)
    print("🤖 Starting corn model training... Please wait.")
    print("=" * 50)
    corn_model.fit(X_corn_train, y_corn_train, epochs = 10, batch_size = 1, verbose = 1, callbacks = [early_stopping])

    # Wheat fit
    print("=" * 50)
    print("🤖 Starting wheat model training... Please wait.")
    print("=" * 50)
    wheat_model.fit(X_wheat_train, y_wheat_train, epochs = 10, batch_size = 1, verbose = 1, callbacks = [early_stopping])

    # Predictions
    print("=" * 50)
    print("📊 Making predictions... Please wait.")
    print("=" * 50)

    soy_train_pred = soy_model.predict(X_soy_train)
    soy_test_pred = soy_model.predict(X_soy_test)
    corn_train_pred = corn_model.predict(X_corn_train)
    corn_test_pred = corn_model.predict(X_corn_test)
    wheat_train_pred = wheat_model.predict(X_wheat_train)
    wheat_test_pred = wheat_model.predict(X_wheat_test)

    # Transform back to original scale to calculate error
    soy_train_pred = soy_scaler.inverse_transform(soy_train_pred)
    y_soy_train_rescaled = soy_scaler.inverse_transform([y_soy_train])
    soy_test_pred = soy_scaler.inverse_transform(soy_test_pred)
    y_soy_test_rescaled = soy_scaler.inverse_transform([y_soy_test])
    corn_train_pred = corn_scaler.inverse_transform(corn_train_pred)
    y_corn_train_rescaled = corn_scaler.inverse_transform([y_corn_train])
    corn_test_pred = corn_scaler.inverse_transform(corn_test_pred)
    y_corn_test_rescaled = corn_scaler.inverse_transform([y_corn_test])
    wheat_train_pred = wheat_scaler.inverse_transform(wheat_train_pred)
    y_wheat_train_rescaled = wheat_scaler.inverse_transform([y_wheat_train])
    wheat_test_pred = wheat_scaler.inverse_transform(wheat_test_pred)
    y_wheat_test_rescaled = wheat_scaler.inverse_transform([y_wheat_test])

    # Calculating the MAE in train and test
    print("=" * 50)
    print("📉 Calculating errors... Please wait.")
    print("=" * 50)

    soy_train_score = mean_absolute_error(y_soy_train_rescaled[0], soy_train_pred[:, 0])
    soy_test_score = mean_absolute_error(y_soy_test_rescaled[0], soy_test_pred[:, 0])
    corn_train_score = mean_absolute_error(y_corn_train_rescaled[0], corn_train_pred[:, 0])
    corn_test_score = mean_absolute_error(y_corn_test_rescaled[0], corn_test_pred[:, 0])
    wheat_train_score = mean_absolute_error(y_wheat_train_rescaled[0], wheat_train_pred[:, 0])
    wheat_test_score = mean_absolute_error(y_wheat_test_rescaled[0], wheat_test_pred[:, 0])

    # Create an index for the original training data, starting at 'look_back' and ending at 'look_back + len(train_rescaled[0])'.
    # This index will be used to associate each training data point with its corresponding year in the original DataFrame.
    original_train_data_index_soy = soybean_data.index[look_back:look_back + len(y_soy_train_rescaled[0])]
    original_train_data_index_corn = corn_data.index[look_back:look_back + len(y_corn_train_rescaled[0])]
    original_train_data_index_wheat = wheat_data.index[look_back:look_back + len(y_wheat_train_rescaled[0])]

    # Create an index for the original test data.
    # Starts from the end of the standardized training data and goes to the end of the standardized test data.
    # The '2 * look_back' is used to adjust the index accordingly.
    original_test_data_index_soy = soybean_data.index[len(y_soy_train_rescaled[0]) + 2 * look_back : len(y_soy_train_rescaled[0]) + 2 * look_back + len(y_soy_test_rescaled[0])]
    original_test_data_index_corn = corn_data.index[len(y_corn_train_rescaled[0]) + 2 * look_back : len(y_corn_train_rescaled[0]) + 2 * look_back + len(y_corn_test_rescaled[0])]
    original_test_data_index_wheat = wheat_data.index[len(y_wheat_train_rescaled[0]) + 2 * look_back : len(y_wheat_train_rescaled[0]) + 2 * look_back + len(y_wheat_test_rescaled[0])]

    # Creates an index for the training predicted values, starting at 'look_back' and ending at 'look_back + len(predicted_train)'.
    # This index will be used to associate each predicted point in the training set with its corresponding year in the original DataFrame.
    predicted_train_data_index_soy = soybean_data.index[look_back:look_back + len(soy_train_pred)]
    predicted_train_data_index_corn = corn_data.index[look_back:look_back + len(corn_train_pred)]
    predicted_train_data_index_wheat = wheat_data.index[look_back:look_back + len(wheat_train_pred)]

    # Creates an index for the predicted values under test.
    predicted_test_data_index_soy = soybean_data.index[len(y_soy_train_rescaled[0]) + 2 * look_back:len(y_soy_train_rescaled[0]) + 2 * look_back+len(soy_test_pred)]
    predicted_test_data_index_corn = corn_data.index[len(y_corn_train_rescaled[0]) + 2 * look_back:len(y_corn_train_rescaled[0]) + 2 * look_back+len(corn_test_pred)]
    predicted_test_data_index_wheat = wheat_data.index[len(y_wheat_train_rescaled[0]) + 2 * look_back:len(y_wheat_train_rescaled[0]) + 2 * look_back+len(wheat_test_pred)]

    # Forecast
    print("=" * 50)
    print("🔮 Generating forecast... Please wait.")
    print("=" * 50)

    # We use the last entry of the original test series to make the next prediction
    last_data_soy = soy_test_sc[-look_back:]
    last_data_soy = np.reshape(last_data_soy, (1, look_back, 1))
    last_data_corn = corn_test_sc[-look_back:]
    last_data_corn = np.reshape(last_data_corn, (1, look_back, 1))
    last_data_wheat = wheat_test_sc[-look_back:]
    last_data_wheat = np.reshape(last_data_wheat, (1, look_back, 1))

    # Prediction with the model (we use the normalized data)
    soy_prediction = soy_model.predict(last_data_soy)
    soy_prediction = soy_prediction[0, 0]
    soy_prediction = soy_scaler.inverse_transform([[soy_prediction]])[0, 0]
    corn_prediction = corn_model.predict(last_data_corn)
    corn_prediction = corn_prediction[0, 0]
    corn_prediction = corn_scaler.inverse_transform([[corn_prediction]])[0, 0]
    wheat_prediction = wheat_model.predict(last_data_wheat)
    wheat_prediction = wheat_prediction[0, 0]
    wheat_prediction = wheat_scaler.inverse_transform([[wheat_prediction]])[0, 0]

    # Variations %
    soy_variation = (soy_prediction - yesterday_soy_price) / yesterday_soy_price
    corn_variation = (corn_prediction - yesterday_corn_price) / yesterday_corn_price
    wheat_variation = (wheat_prediction - yesterday_wheat_price) / yesterday_wheat_price

    return {
        "soy_train_score": soy_train_score,
        "soy_test_score": soy_test_score,
        "corn_train_score": corn_train_score,
        "corn_test_score": corn_test_score,
        "wheat_train_score": wheat_train_score,
        "wheat_test_score": wheat_test_score,
        "soy_prediction": soy_prediction,
        "corn_prediction": corn_prediction,
        "wheat_prediction": wheat_prediction,
        "soy_variation": soy_variation,
        "corn_variation": corn_variation,
        "wheat_variation": wheat_variation,
        "original_test_data_index_soy": original_test_data_index_soy,
        "original_test_data_index_corn": original_test_data_index_corn,
        "original_test_data_index_wheat": original_test_data_index_wheat,
        "predicted_test_data_index_soy": predicted_test_data_index_soy,
        "predicted_test_data_index_corn": predicted_test_data_index_corn,
        "predicted_test_data_index_wheat": predicted_test_data_index_wheat,
        "y_soy_test_rescaled": y_soy_test_rescaled[0],
        "y_corn_test_rescaled": y_corn_test_rescaled[0],
        "y_wheat_test_rescaled": y_wheat_test_rescaled[0],
        "soy_test_pred": soy_test_pred[:, 0],
        "corn_test_pred": corn_test_pred[:, 0],
        "wheat_test_pred": wheat_test_pred[:, 0],
    }

results = training_module()

print("=" * 50)
print("✅ Forecast module completed!")
print("=" * 50)


#################################### Agents AI module ###########################################

print("=" * 50)
print("🤖 Building the agents!")
print("=" * 50)

# Decorator that allows caching of resources in Streamlit to improve application performance
@st.cache_resource
def building_agents():

    # Loading enviroment
    load_dotenv()

    # Agent for climate and agricultural conditions
    climate_agent = Agent(
        name="Climate Intelligence Agent",
        role="Climatic and Weather Conditions",
        model=Groq(id="deepseek-r1-distill-llama-70b"),
        tools=[DuckDuckGo()],
        instructions=[
            "Search for recent and forecasted weather events affecting major agricultural regions (e.g., Brazil, USA, Argentina).",
            "Identify droughts, floods, heatwaves, or planting/harvest delays that might impact commodity supply.",
        ],
        show_tool_calls=True,
        markdown=True
    )

    # Agent for recommendations and market sentiment
    sentiment_agent = Agent(
        name="Analyst Sentiment Agent",
        role="Market Sentiment and Analyst Opinions",
        model=Groq(id="deepseek-r1-distill-llama-70b"),
        tools=[DuckDuckGo()],
        instructions=[
            "Search for analyst recommendations and market sentiment on commodities.",
            "The market are bullish or bearish about the commoditie?.",
        ],
        show_tool_calls=True,
        markdown=True
    )


    # Integrating agent to consolidate analyses
    commodity_analysis_agent = Agent(
        team=[climate_agent, sentiment_agent],
        model=Groq(id="llama-3.3-70b-versatile"),
        instructions=[
            "Summarize how current weather conditions, market sentiment, and financial data in the last days are impacting commodity prices. Bring the information into topics.",
            "Always include sources and links at the end of the report."
        ],
        show_tool_calls=True,
        markdown=True
    )

    # List of commodities
    commodities = [
        {"name": "soybean"},
        {"name": "corn"},
        {"name": "wheat"}
    ]

    print("=" * 50)
    print("🔮 The agents are working... Please wait.")
    print("=" * 50)

    # Dictionary to store the final AI outputs
    commodity_responses = {}

    # Loop through each commodity to generate a tailored analysis
    for commodity in commodities:
        name = commodity["name"]

        # Detailed prompt for the agent team
        prompt = (f"Analyze the commodity {name.upper()}.")

        # Run the specialized AI agent team
        ai_response = commodity_analysis_agent.run(prompt)

        # Clean the response: remove "Running:" blocks and transfer logs
        clean_response = re.sub(r"(Running:[\s\S]*?\n\n)", "", ai_response.content, flags=re.MULTILINE).strip()

        # Store the cleaned response
        commodity_responses[name] = clean_response

    soybean_agent = commodity_responses["soybean"]
    corn_agent = commodity_responses["corn"]
    wheat_agent = commodity_responses["wheat"]

    return soybean_agent, corn_agent, wheat_agent

soybean_agent_report, corn_agent_report, wheat_agent_report = building_agents()

#################################### Charts code ########################################## 

# MACD
def plot_macd(data):
    last_100 = data.tail(100)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['macd'], mode='lines', name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['macd_signal_line'], mode='lines', name='Signal', line=dict(color='magenta')))
    fig.add_trace(go.Bar(x=last_100.index, y=last_100['macd_diff'], name='Histogram', marker_color='black', opacity=0.8))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title='MACD Value',
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=0.5))
    return fig


# Bollinger bands
def plot_bollinger_bands(data):
    last_100 = data.tail(100)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['Close'], mode='lines', name='Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['bb_high'], mode='lines', name='Upper Band', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['bb_low'], mode='lines', name='Lower Band', line=dict(color='green')))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title='Price',
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=0.5))
    return fig

# Return
def plot_returns(data):
    last_100 = data.tail(100)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['returns'], mode='lines', name='Returns', line=dict(color='blue')))
    fig.update_layout(height=400, 
                      margin=dict(l=20, r=20, t=20, b=20),
                      yaxis_title='Return')
    return fig

# RSI
def plot_rsi(data):
    last_100 = data.tail(100)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['rsi'], mode='lines', name='RSI', line=dict(color='blue')))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title='RSI')
    return fig

def plot_close(data):
    last_100 = data.tail(100)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=last_100.index, y=last_100['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title='Close')
    return fig

#################################### App module ###########################################

# Streamlit application
print("=" * 50)
print("🔮 The aplication is being generated... Please wait.")
print("=" * 50)

# Title
st.title("Trader's assistant for daily commodity price analysis")

# Adding sidebar content
st.sidebar.title("Welcome!")
st.sidebar.markdown("""
This application was developed to support decision-making in commodity trading, with a focus on **soybean**, **wheat**, and **corn**.

The assistant performs the following tasks:

- 📊 **Real-time Data Extraction**  
  Automatically collects up-to-date market data for key agricultural commodities.

- 🔮 **Next-Day Price Prediction**  
  Uses analytical models to forecast price movements for the next trading day.

- 📉 **Technical Indicators Charts**  
  Displays key technical analysis indicators (e.g. MACD, RSI, Bollinger Bands) to support trading strategies.

- 🌦️ **Weather & Sentiment Monitoring**  
  Gathers recent weather updates and analyzes market sentiment that may influence prices.

All these features are integrated to provide a smarter and more informed trading experience.
""")

# Adding white space to adjust the size of the sidebar
st.sidebar.markdown("&#8203;" * 100)

# Page selection module
def select_page():

    # Creates a sidebar menu with selection options
    option = st.sidebar.selectbox('Select the commodity', ['Soybean', 'Corn', 'Wheat'])
    
    # Checks if the selected option is 'View Charts' and calls the corresponding function
    if option == 'Soybean':
        soybean_page()
        
    # Checks if the selected option is 'View Data Table' and calls the corresponding function
    elif option == 'Corn':
        corn_page()
        
    # If none of the above options is selected, assumes the option is 'Make Predictions' and calls the corresponding function
    else:
        wheat_page()

########## Soybean page ##############
# Show soybean page on streamlit
def soybean_page():

    # Create a header on the page
    st.header('Soybean Report')
    st.write("---")

    # Print predictions
    st.header("🔮 Daily Soybean Predictions")
    st.write(f"- Predicted price for today: **{results["soy_prediction"]:.2f}**")
    st.write(f"- Yesterday's price: **{yesterday_soy_price:.2f}**")
    st.write(f"- Variation: **{results["soy_variation"] * 100:.2f}%** compared to the previous day")
    st.write(f"- MAE error on training set: {results["soy_train_score"]:.4f}")
    st.write(f"- MAE error on test set: {results["soy_test_score"]:.4f}")
    st.write("---")

    # Predicted vs Actual Price chart in test data
    st.header("📊 Predicted VS Real Values in Test Data")
    # Plotly Chart
    fig_soy = go.Figure()
    fig_soy.add_trace(go.Scatter(
        x=pd.to_datetime(results["original_test_data_index_soy"]), y=results["y_soy_test_rescaled"],
        mode='lines',
        name='Actual Soybean Prices',
        line=dict(color='green', dash='solid')))
    fig_soy.add_trace(go.Scatter(
        x=pd.to_datetime(results["predicted_test_data_index_soy"]), y=results["soy_test_pred"],
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red', dash='dash')))
    fig_soy.update_layout(
        title='Soybean: Actual vs. Predicted',
        yaxis_title='Price',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)', bordercolor='black', borderwidth=0.5))
    # Show the chart
    st.plotly_chart(fig_soy, use_container_width=True)

    st.write("---")

    # Technical Indicators Charts
    st.header("📈 Technical Indicators Charts")

    # Create a selection widget to choose a technical indicator
    option = st.radio('Select a Technical Indicator to View',[ 'Closing Price', 
                                                               'Bollinger Bands', 
                                                               'MACD', 
                                                               'Returns', 
                                                               'RSI'])
    if option == 'Closing Price':
        st.write('**Closing Price (Last 100 Days)**')
        st.markdown("""
        The closing price chart displays the final price at which the asset traded on each day. 
        It provides a basic view of price movement over time and is commonly used to assess overall trends.
        """)
        st.plotly_chart(plot_close(soybean_data), use_container_width=True)

    elif option == 'Bollinger Bands':
        st.write('**Bollinger Bands (14-Day Window, Last 100 Days)**')
        st.markdown("""
        Bollinger Bands consist of a moving average, an upper band, and a lower band.
        They reflect volatility: bands widen in high volatility and contract in low volatility.
        When the price approaches the upper band, the asset may be overbought; 
        when it nears the lower band, it may be oversold.
        """)
        st.plotly_chart(plot_bollinger_bands(soybean_data), use_container_width=True)

    elif option == 'MACD':
        st.write('**MACD - Moving Average Convergence Divergence (Last 100 Days)**')
        st.markdown("""
        The MACD helps identify momentum shifts and trend direction.
        - The MACD line is the difference between two EMAs.
        - The Signal line is an EMA of the MACD line.
        - The Histogram shows the distance between the MACD and Signal lines.
        
        When the MACD crosses above the signal line, it may signal a bullish trend; 
        crossing below can indicate a bearish trend.
        """)
        st.plotly_chart(plot_macd(soybean_data), use_container_width=True)

    elif option == 'Returns':
        st.write('**Daily Returns (Last 100 Days)**')
        st.markdown("""
        This chart shows the daily percentage change in closing price.
        It reflects volatility and helps understand day-to-day fluctuations.
        Positive returns indicate price gains; negative returns reflect losses.
        Spikes or dips may suggest market reactions to news or events.
        """)
        st.plotly_chart(plot_returns(soybean_data), use_container_width=True)

    else:
        st.write('**RSI - Relative Strength Index (14-Day Window, Last 100 Days)**')
        st.markdown("""
        The RSI measures the speed and change of price movements, scaled from 0 to 100.
        - RSI > 70: Asset may be overbought (potential sell signal).
        - RSI < 30: Asset may be oversold (potential buy signal).
        
        It helps identify potential reversal points in the market.
        """)
        st.plotly_chart(plot_rsi(soybean_data), use_container_width=True)

    st.write("---")

    # AI Agents Analysis
    st.header("👾 AI Agents Analysis")
    st.markdown(soybean_agent_report)


########## Corn page ##############
# Show Corn page on streamlit
def corn_page():

    # Create a header on the page
    st.header('Corn Report')
    st.write("---")

    # Print predictions
    st.header("🔮 Daily Corn Predictions")
    st.write(f"- Predicted price for today: **{results["corn_prediction"]:.2f}**")
    st.write(f"- Yesterday's price: **{yesterday_corn_price:.2f}**")
    st.write(f"- Variation: **{results["corn_variation"] * 100:.2f}%** compared to the previous day")
    st.write(f"- MAE error on training set: {results["corn_train_score"]:.4f}")
    st.write(f"- MAE error on test set: {results["corn_test_score"]:.4f}")
    st.write("---")

    # Predicted vs Actual Price chart in test data
    st.header("📊 Predicted Values VS Real Values in Test Data")
    # Plotly Chart
    fig_corn = go.Figure()
    fig_corn.add_trace(go.Scatter(
        x=pd.to_datetime(results["original_test_data_index_corn"]), y=results["y_corn_test_rescaled"],
        mode='lines',
        name='Actual Corn Prices',
        line=dict(color='green', dash='solid')))
    fig_corn.add_trace(go.Scatter(
        x=pd.to_datetime(results["predicted_test_data_index_corn"]), y=results["corn_test_pred"],
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red', dash='dash')))
    fig_corn.update_layout(
        title='Corn: Actual vs. Predicted',
        yaxis_title='Price',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)', bordercolor='black', borderwidth=0.5))
    # Show the chart
    st.plotly_chart(fig_corn, use_container_width=True)

    st.write("---")

    # Technical Indicators Charts
    st.header("📈 Technical Indicators Charts")

    # Create a selection widget to choose a technical indicator
    option = st.radio('Select a Technical Indicator to View',['Closing Price', 
                                                               'Bollinger Bands', 
                                                               'MACD', 
                                                               'Returns', 
                                                               'RSI'])
    if option == 'Closing Price':
        st.write('**Closing Price (Last 100 Days)**')
        st.markdown("""
        The closing price chart displays the final price at which the asset traded on each day. 
        It provides a basic view of price movement over time and is commonly used to assess overall trends.
        """)
        st.plotly_chart(plot_close(corn_data), use_container_width=True)

    elif option == 'Bollinger Bands':
        st.write('**Bollinger Bands (14-Day Window, Last 100 Days)**')
        st.markdown("""
        Bollinger Bands consist of a moving average, an upper band, and a lower band.
        They reflect volatility: bands widen in high volatility and contract in low volatility.
        When the price approaches the upper band, the asset may be overbought; 
        when it nears the lower band, it may be oversold.
        """)
        st.plotly_chart(plot_bollinger_bands(corn_data), use_container_width=True)

    elif option == 'MACD':
        st.write('**MACD - Moving Average Convergence Divergence (Last 100 Days)**')
        st.markdown("""
        The MACD helps identify momentum shifts and trend direction.
        - The MACD line is the difference between two EMAs.
        - The Signal line is an EMA of the MACD line.
        - The Histogram shows the distance between the MACD and Signal lines.
        
        When the MACD crosses above the signal line, it may signal a bullish trend; 
        crossing below can indicate a bearish trend.
        """)
        st.plotly_chart(plot_macd(corn_data), use_container_width=True)

    elif option == 'Returns':
        st.write('**Daily Returns (Last 100 Days)**')
        st.markdown("""
        This chart shows the daily percentage change in closing price.
        It reflects volatility and helps understand day-to-day fluctuations.
        Positive returns indicate price gains; negative returns reflect losses.
        Spikes or dips may suggest market reactions to news or events.
        """)
        st.plotly_chart(plot_returns(corn_data), use_container_width=True)

    else:
        st.write('**RSI - Relative Strength Index (14-Day Window, Last 100 Days)**')
        st.markdown("""
        The RSI measures the speed and change of price movements, scaled from 0 to 100.
        - RSI > 70: Asset may be overbought (potential sell signal).
        - RSI < 30: Asset may be oversold (potential buy signal).
        
        It helps identify potential reversal points in the market.
        """)
        st.plotly_chart(plot_rsi(corn_data), use_container_width=True)

    st.write("---")

    # AI Agents Analysis
    st.header("👾 AI Agents Analysis")
    st.markdown(corn_agent_report)


########## Wheat page ##############
# Show Wheat page on streamlit
def wheat_page():

    # Create a header on the page
    st.header('Wheat Report')
    st.write("---")

    # Print predictions
    st.header("🔮 Daily Wheat Predictions")
    st.write(f"- Predicted price for today: **{results["wheat_prediction"]:.2f}**")
    st.write(f"- Yesterday's price: **{yesterday_wheat_price:.2f}**")
    st.write(f"- Variation: **{results["wheat_variation"] * 100:.2f}%** compared to the previous day")
    st.write(f"- MAE error on training set: {results["wheat_train_score"]:.4f}")
    st.write(f"- MAE error on test set: {results["wheat_test_score"]:.4f}")
    st.write("---")

   # Predicted vs Actual Price chart in test data
    st.header("📊 Predicted Values VS Real Values in Test Data")
    # Plotly Chart
    fig_wheat = go.Figure()
    fig_wheat.add_trace(go.Scatter(
        x=pd.to_datetime(results["original_test_data_index_wheat"]), y=results["y_wheat_test_rescaled"],
        mode='lines',
        name='Actual Wheat Prices',
        line=dict(color='green', dash='solid')))
    fig_wheat.add_trace(go.Scatter(
        x=pd.to_datetime(results["predicted_test_data_index_wheat"]), y=results["wheat_test_pred"],
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red', dash='dash')))
    fig_wheat.update_layout(
        title='Wheat: Actual vs. Predicted',
        yaxis_title='Price',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.5)', bordercolor='black', borderwidth=0.5))
    # Show the chart
    st.plotly_chart(fig_wheat, use_container_width=True)
    
    st.write("---")

    # Technical Indicators Charts
    st.header("📈 Technical Indicators Charts")

    # Create a selection widget to choose a technical indicator
    option = st.radio('Select a Technical Indicator to View',['Closing Price', 
                                                               'Bollinger Bands', 
                                                               'MACD', 
                                                               'Returns', 
                                                               'RSI'])
    if option == 'Closing Price':
        st.write('**Closing Price (Last 100 Days)**')
        st.markdown("""
        The closing price chart displays the final price at which the asset traded on each day. 
        It provides a basic view of price movement over time and is commonly used to assess overall trends.
        """)
        st.plotly_chart(plot_close(wheat_data), use_container_width=True)

    elif option == 'Bollinger Bands':
        st.write('**Bollinger Bands (14-Day Window, Last 100 Days)**')
        st.markdown("""
        Bollinger Bands consist of a moving average, an upper band, and a lower band.
        They reflect volatility: bands widen in high volatility and contract in low volatility.
        When the price approaches the upper band, the asset may be overbought; 
        when it nears the lower band, it may be oversold.
        """)
        st.plotly_chart(plot_bollinger_bands(wheat_data), use_container_width=True)

    elif option == 'MACD':
        st.write('**MACD - Moving Average Convergence Divergence (Last 100 Days)**')
        st.markdown("""
        The MACD helps identify momentum shifts and trend direction.
        - The MACD line is the difference between two EMAs.
        - The Signal line is an EMA of the MACD line.
        - The Histogram shows the distance between the MACD and Signal lines.
        
        When the MACD crosses above the signal line, it may signal a bullish trend; 
        crossing below can indicate a bearish trend.
        """)
        st.plotly_chart(plot_macd(wheat_data), use_container_width=True)

    elif option == 'Returns':
        st.write('**Daily Returns (Last 100 Days)**')
        st.markdown("""
        This chart shows the daily percentage change in closing price.
        It reflects volatility and helps understand day-to-day fluctuations.
        Positive returns indicate price gains; negative returns reflect losses.
        Spikes or dips may suggest market reactions to news or events.
        """)
        st.plotly_chart(plot_returns(wheat_data), use_container_width=True)

    else:
        st.write('**RSI - Relative Strength Index (14-Day Window, Last 100 Days)**')
        st.markdown("""
        The RSI measures the speed and change of price movements, scaled from 0 to 100.
        - RSI > 70: Asset may be overbought (potential sell signal).
        - RSI < 30: Asset may be oversold (potential buy signal).
        
        It helps identify potential reversal points in the market.
        """)
        st.plotly_chart(plot_rsi(wheat_data), use_container_width=True)

    st.write("---")

    # AI Agents Analysis
    st.header("👾 AI Agents Analysis")
    st.markdown(wheat_agent_report)

# Main block of Python program
if __name__ == '__main__':
        select_page()










