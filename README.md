# 🌾 Commodities Trade Application

*An intelligent trading assistant for soybean, corn, and wheat — powered by LSTM forecasts, technical analysis, and real-time market insights from AI agents.*

## 📌 Overview

This project is an intelligent trading assistant designed to help traders and market analysts forecast next-day prices of key agricultural commodities — soybean, corn, and wheat. It addresses the challenges of price volatility and market unpredictability by leveraging advanced machine learning techniques.

The target users are individual traders, commodity market analysts, and agricultural businesses seeking data-driven insights to optimize their trading strategies and decision-making processes.

The system combines state-of-the-art technologies such as Long Short-Term Memory (LSTM) networks for time series forecasting, interactive technical analysis charts, and AI-powered agents that gather real-time news and sentiment data relevant to commodity markets.

The application runs locally using Docker, enabling users to start a containerized environment that hosts a web-based dashboard built with Streamlit. This dashboard presents forecasts, technical indicators, and up-to-date market intelligence to support informed trading decisions.

## 🔍 Features

- **Price Forecasting with LSTM:**  
  Utilizes a Long Short-Term Memory (LSTM) model to predict next-day closing prices of commodities. The model employs a look-back window of 10 days, using only the 'Close' price from historical data to forecast future values.

- **Technical Analysis Charts:**  
  Provides interactive charts for key technical indicators including Returns, Bollinger Bands, MACD, RSI, SMA, EMA, ROC, and Close prices to support trading decisions.

- **AI Agents for Market Insights:**  
  - *Climate Intelligence Agent:* Uses the Groq deepseek-r1-distill-llama-70b model combined with DuckDuckGo search API to gather and analyze recent and forecasted weather events affecting major agricultural regions (e.g., droughts, floods, heatwaves) that impact commodity supply.  
  - *Analyst Sentiment Agent:* Employs the same Groq model and DuckDuckGo to collect analyst recommendations and market sentiment, identifying bullish or bearish trends for commodities.  
  - *Commodity Analysis Agent:* Integrates insights from climate and sentiment agents using the llama-3.3-70b-versatile model to summarize how weather, market sentiment, and recent financial data affect commodity prices, providing comprehensive reports with sources and links.  
  These models leverage free API calls sufficient for project needs.

- **Dockerized Deployment:**  
  The entire application is containerized with Docker for simplified setup, consistent environments, and easy deployment across different machines without dependency issues.

## 🧱 Project Architecture

The project is divided into two main modules that work together to deliver the complete application:

- **Data Extraction:** responsible for collecting and preparing historical commodity data. This module runs a Python script that extracts the data and saves it in a shared Docker volume.

- **Commodities Application:** the main application that includes the LSTM model for price prediction, generation of technical analysis charts, and AI agents for climate and market sentiment analysis. This part is a web app built with Streamlit that accesses the data prepared by the extraction module.

Communication between the modules happens through a Docker volume named `commodities_trade_app`, where the data is stored for use by the application.

The project structure is as follows:

```
commodities-trade-app/
├── app/ # Frontend with Streamlit
│ ├── commodities_app.py
│ ├── requirements.txt
│ ├── Dockerfile
│ └── .env # API key from groq.com
├── data_extraction/ # Module responsible for extracting and preparing data
│ ├── data_extraction.py
│ ├── requirements.txt
│ └── Dockerfile
└── docker-compose.yml # Service orchestration
```

## 🐳 Running with Docker

To run the Commodities Trade Application locally, you only need to have Docker Desktop installed on your computer.

Follow these steps:

1. Open the command prompt (Windows) or terminal (Mac/Linux).
   
2. Navigate to the project folder where the files are located using the `cd` command. For example:
   
   **cd path\to\commodities-trade-app**
   
3. Run the following command to build and start the Docker containers:
   
   **docker-compose up --build**

4. Once running, open your browser and go to:
 
   **http://localhost:8501**


## 🧪 Application Demonstration

Below is a quick demonstration of the Commodities Trade Application in action.  
It showcases the main features including price prediction, technical indicators, and AI agent insights — all accessible through the Streamlit interface.

[![Commodities Trade App Demo](path/to/your/demo.gif)](https://github.com/user-attachments/assets/c9291bbf-d15d-4ee2-9d87-35f008b4691d)


## 🤝 Contributing

Contributions are welcome! If you have suggestions to improve this project, feel free to share!
All feedback, improvements, and ideas are highly appreciated!


## 🌐 Connect with Me

- GitHub: https://github.com/JoaoZ2001
- LinkedIn: https://www.linkedin.com/in/joaovzimmermann

If you liked this project, consider giving it a ⭐ to help others find it!









