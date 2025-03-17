# MVO+CDA Simulation tool

## 📌 Project Overview
This project is a **portfolio simulation tool** that allows users to test different investment strategies using historical ETF data. The tool utilizes financial optimization techniques to allocate capital dynamically and analyze investment performance over time.

## 🎯 Key Features
- **Mean-Variance Optimization (MVO)**: Dynamically adjusts investment allocation based on risk and return.
- **Dollar-Cost Averaging (DCA)**: Simulates a fixed weekly investment strategy.
- **SPY Benchmark Strategy**: Compares MVO and DCA performance against a simple SPY (S&P 500) strategy.
- **Historical Data Analysis**: Fetches and processes ETF price data from Yahoo Finance.
- **Portfolio Visualization**: Generates plots of portfolio growth, asset distribution, and risk metrics.
- **CSV Export**: Saves investment logs for further analysis.

## ⚙️ Technologies Used
- **Python** (NumPy, Pandas, Matplotlib, CVXPY, Yahoo Finance API)
- **Data Analysis & Optimization**: Mean-Variance Portfolio Theory
- **Visualization**: Matplotlib, Plotly (planned)
- **Web Deployment (Planned)**: Streamlit for interactive user interface

## 📊 Investment Strategies Implemented
### 1️⃣ **Mean-Variance Optimization (MVO)**
A dynamic asset allocation strategy based on Markowitz's Modern Portfolio Theory:
- Minimizes portfolio risk while maintaining expected returns.
- Allocates capital weekly based on historical price data.
- Ensures no short-selling and weight constraints.

### 2️⃣ **Dollar-Cost Averaging (DCA)**
A passive investment strategy that spreads investments evenly over time:
- Fixed weekly investment into selected ETFs.
- Helps reduce volatility risk by averaging purchase prices.
- Simple but effective long-term strategy.

### 3️⃣ **SPY Benchmark**
A comparison model investing only in **SPY ETF**, simulating a passive S&P 500 investment:
- Allows performance benchmarking against broader market indices.

## 📈 Planned Improvements
- **Interactive Web Dashboard** (Streamlit) 🖥️
- **Expanded Portfolio Analysis** (Sharpe Ratio, Max Drawdown, CAGR)
- **Alternative Asset Classes** (Cryptos, Bonds, Commodities)
- **Backtesting with Macro Factors** (Interest rates, Inflation)

## 🚀 How to Run the Simulation
1. Install dependencies:
   ```sh
   pip install numpy pandas matplotlib cvxpy yfinance
   ```
2. Run the script:
   ```sh
   python invest.py
   ```
3. View the generated results in CSV or visual plots.

---
This is the foundation of an **interactive investment simulator**, with planned upgrades to enhance usability and real-world applicability.

