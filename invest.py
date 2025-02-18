import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
import matplotlib.pyplot as plt

# 🏆 List of ETFs to optimize
ETF_LIST = ["QQQ", "SPY", "VOO", "EEM", "VWO", "QQQM","IAU"]
WEEKLY_BUDGET = 800  # Fixed investment per week
START_DATE = "2022-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

def fetch_data():
    """Fetches historical weekly price data for ETFs from Yahoo Finance"""
    prices = yf.download(ETF_LIST, start=START_DATE, end=END_DATE, interval="1wk")["Close"]
    return prices

def compute_returns(prices):
    """Computes weekly percentage returns"""
    returns = prices.pct_change().dropna()
    return returns

def make_covariance_psd(cov_matrix, epsilon=1e-4):
    """Ensures the covariance matrix is positive semi-definite (PSD)"""
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Compute eigenvalues
    eigvals = np.maximum(eigvals, epsilon)  # Replace negative eigenvalues with a small positive number
    return eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct matrix

def mean_variance_optimization(returns):
    """Computes optimal allocation weights using Mean-Variance Optimization (MVO)"""
    num_assets = returns.shape[1]

    # **FIX**: Ensure we have enough data
    if len(returns) < 10:  # Require at least 10 weeks of data
        return np.full(num_assets, 1/num_assets)  # Equal weights if not enough data

    # Compute mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # **FIX**: Ensure covariance matrix is positive semi-definite (PSD)
    cov_matrix = make_covariance_psd(cov_matrix, epsilon=1e-4)

    # Define optimization variables
    w = cp.Variable(num_assets)  # Allocation weights
    risk = cp.quad_form(w, cov_matrix)  # Portfolio variance (quadratic form)

    # Optimization problem: Minimize risk with constraints
    problem = cp.Problem(cp.Minimize(risk), [
        cp.sum(w) == 1,  # Weights must sum to 1
        w >= 0  # No short-selling
    ])

    problem.solve()

    weights = np.array(w.value) if w.value is not None else np.full(num_assets, 1/num_assets)

    # **FIX**: Explicitly normalize weights to sum exactly to 1
    weights /= np.sum(weights)

    return weights

def simulate_mvo_dca():
    """Simulates a weekly investment strategy using MVO for dynamic allocation"""
    prices = fetch_data()
    returns = compute_returns(prices)
    portfolio = {etf: 0 for etf in ETF_LIST}
    investment_log = []

    for date, row in prices.iterrows():
        if date not in returns.index:
            continue

        returns_so_far = returns.loc[:date].dropna()

        optimal_weights = mean_variance_optimization(returns_so_far)

        raw_allocations = WEEKLY_BUDGET * np.array(optimal_weights)

        rounded_allocations = np.round(raw_allocations, 2)

        difference = WEEKLY_BUDGET - np.sum(rounded_allocations)

        if difference != 0:
            max_index = np.argmax(raw_allocations)
            rounded_allocations[max_index] += difference

        allocation = rounded_allocations


        for i, etf in enumerate(ETF_LIST):
            price = row[etf]

            if np.isnan(price) or allocation[i] <= 0:
                continue

            shares = round(allocation[i] / price, 4)

            portfolio[etf] += shares

            investment_log.append([date.strftime('%Y-%m-%d'), etf, price, shares, allocation[i] ])

    df = pd.DataFrame(investment_log, columns=["Date", "ETF", "Price", "Shares", "Total_Invested"])
    df.to_csv("mvo_dca_simulation_fixed.csv", index=False)

    print("✅ MVO + DCA Simulation Complete. Results saved to mvo_dca_simulation_fixed.csv")
    return df, portfolio

def plot_portfolio_value(df, portfolio):
    """Plots the total portfolio value over time"""
    prices = fetch_data()
    portfolio_values = []

    for date, row in prices.iterrows():
        value = sum(portfolio[etf] * row[etf] for etf in ETF_LIST if not np.isnan(row[etf]))
        portfolio_values.append([date.strftime('%Y-%m-%d'), value])

    df_value = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio_Value"])
    df_value["Date"] = pd.to_datetime(df_value["Date"])

    # 📈 Plot portfolio value over time
    plt.figure(figsize=(10,5))
    plt.plot(df_value["Date"], df_value["Portfolio_Value"], label="Portfolio Value (MVO + DCA)", color="b")
    plt.xlabel("Date")
    plt.ylabel("Value in USD")
    plt.title("MVO + DCA Portfolio Value Over Time (Fixed)")
    plt.legend()
    plt.grid()
    plt.show()

def show_portfolio_holdings(portfolio, prices):
    """
    Displays the portfolio holdings as a pie chart and prints total portfolio value.

    Parameters:
    - portfolio: Dictionary with ETF names as keys and number of shares as values.
    - prices: DataFrame with the latest ETF prices.
    """
    latest_prices = prices.iloc[-1]  # Get the latest available prices
    print(prices)
    portfolio_values = {etf: portfolio[etf] * latest_prices[etf] for etf in portfolio if not np.isnan(latest_prices[etf])}

    total_value = sum(portfolio_values.values())

    # Print total portfolio value
    print(f"💰 Total Portfolio Value: ${total_value:.2f}")

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(portfolio_values.values(), labels=portfolio_values.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Holdings Distribution")
    plt.show()

def simulate_spy_dca():
    """Simulates a weekly investment strategy into SPY (S&P 500 ETF) using Dollar-Cost Averaging (DCA)."""
    prices = yf.download("SPY", start=START_DATE, end=END_DATE, interval="1wk")["Close"]
    prices = prices.dropna() 

    portfolio = {"SPY": 0}  # Only SPY investments
    investment_log = []

    for date, price in prices.iterrows():
        price = price[0]
        if np.isnan(price) or price <= 0:
            continue

        shares = round(WEEKLY_BUDGET / price, 4)
        portfolio["SPY"] += shares

        investment_log.append([date.strftime('%Y-%m-%d'), "SPY", price, shares, WEEKLY_BUDGET])

    df = pd.DataFrame(investment_log, columns=["Date", "ETF", "Price", "Shares", "Total_Invested"])
    df.to_csv("spy_dca_simulation.csv", index=False)

    print("✅ SPY DCA Simulation Complete. Results saved to spy_dca_simulation.csv")
    return df, portfolio


def plot_spy_portfolio_value(df, portfolio):
    """Plots the total portfolio value over time for SPY DCA strategy."""
    prices = yf.download("SPY", start=START_DATE, end=END_DATE, interval="1wk")["Close"]
    prices = prices.dropna()

    portfolio_values = []

    for date, price in prices.iterrows():
        value = portfolio["SPY"] * price
        portfolio_values.append([date.strftime('%Y-%m-%d'), value])

    df_value = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio_Value"])
    df_value["Date"] = pd.to_datetime(df_value["Date"])

    # 📈 Plotting SPY values
    plt.figure(figsize=(10, 5))
    plt.plot(df_value["Date"], df_value["Portfolio_Value"], label="Portfolio Value (SPY DCA)", color="r", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Value in USD")
    plt.title("SPY DCA Portfolio Value Over Time")
    plt.legend()
    plt.grid()
    plt.show()


# Run the simulation
df, portfolio = simulate_mvo_dca()
plot_portfolio_value(df, portfolio)
show_portfolio_holdings(portfolio, fetch_data())
df_spy, portfolio_spy = simulate_spy_dca()
plot_spy_portfolio_value(df_spy, portfolio_spy)
show_portfolio_holdings(portfolio_spy, fetch_data())