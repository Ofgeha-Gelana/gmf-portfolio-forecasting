# **Time Series Forecasting and Portfolio Optimization for Guide Me in Finance (GMF) Investments**  

This project leverages advanced time series forecasting techniques to enhance portfolio management strategies for **Guide Me in Finance (GMF) Investments**. By analyzing historical financial data from **YFinance**, we aim to predict market trends, optimize asset allocation, and develop investment strategies that maximize returns while minimizing risk.  

## **Project Overview**  

### **Objective**  
Develop and evaluate time series forecasting models to predict stock trends, providing data-driven insights for portfolio optimization and risk management.  

### **Key Assets**  
The portfolio consists of a mix of assets to balance risk and return:  
- **TSLA (Tesla)** – High-growth, high-risk stock.  
- **BND (Vanguard Total Bond Market ETF)** – Stability-focused bond ETF.  
- **SPY (S&P 500 ETF)** – Broad market exposure for diversification.  

## **Methodology**  

### **1. Data Collection & Preprocessing**  
- **Data Sources** – Extracted historical data (2015–2024) for TSLA, BND, and SPY using YFinance.  
- **Data Cleaning** – Handled missing values, ensured data consistency, and adjusted for stock splits and dividends.  
- **Exploratory Data Analysis (EDA)** – Identified trends, volatility, and seasonality using statistical methods, rolling averages, and outlier detection.  

### **2. Time Series Forecasting Models**  
Developed and compared multiple forecasting models:  
- **ARIMA** – Suitable for univariate, non-seasonal time series forecasting.  
- **SARIMA** – Extended ARIMA to account for seasonality in financial data.  
- **LSTM (Long Short-Term Memory Networks)** – Deep learning model to capture long-term dependencies in stock price movements.  

#### **Model Evaluation**  
- **Metrics Used**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).  
- **Forecast Horizon**: Predicted stock price movements for 6 to 12 months.  
- **Forecast Visualization**: Displayed trends with confidence intervals to assess expected market movements, volatility, and potential risks.  

### **3. Portfolio Optimization**  
- **Portfolio Composition** – Balanced TSLA, BND, and SPY to optimize risk-return tradeoff.  
- **Risk & Return Analysis**  
  - **Annual Returns** – Projected expected returns based on forecasted trends.  
  - **Covariance & Volatility** – Evaluated asset correlations and historical volatility.  
  - **Sharpe Ratio Optimization** – Determined optimal asset weights to maximize risk-adjusted returns.  

## **Conclusion**  
This project integrates **time series forecasting** and **portfolio optimization** to provide actionable investment insights for GMF Investments. By leveraging predictive analytics, the firm can proactively adjust asset allocations, mitigate risks, and enhance financial performance.  

## **Installation & Setup**  

### **Requirements**  
Ensure you have Python installed along with the required libraries:  

```bash
pip install yfinance numpy pandas matplotlib scikit-learn statsmodels tensorflow
```

### **Usage**  
Run the main script to fetch data, train models, and generate forecasts:  

```bash
python main.py
```

### **Project Structure**  
```
📂 GMF_TimeSeries_Forecasting
│── data/                 # Historical stock data
│── models/               # Trained forecasting models
│── notebooks/            # Jupyter Notebooks for EDA and model training
│── src/
│   │── data_loader.py    # Fetch and preprocess data
│   │── model_train.py    # Train forecasting models
│   │── portfolio_opt.py  # Optimize portfolio allocation
│── main.py               # Main execution script
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation
```

### **License**  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
