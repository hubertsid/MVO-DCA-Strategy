import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

ETF_LIST = ["QQQ", "SPY", "VOO", "EEM", "VWO", "QQQM", "IAU"]
WEEKLY_BUDGET = 20
START_DATE = "2022-01-01"
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

def fetch_data():
    """Fetches historical ETF price data from CSV."""
    return pd.read_csv("etf_data.csv", index_col=0, parse_dates=True)

def compute_returns(prices):
    """Computes weekly percentage returns."""
    return prices.pct_change().dropna()

def make_covariance_psd(cov_matrix, epsilon=1e-4):
    """Ensures the covariance matrix is positive semi-definite (PSD)."""
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def mean_variance_optimization(returns):
    """Computes optimal allocation weights using Mean-Variance Optimization (MVO)."""
    num_assets = returns.shape[1]
    if len(returns) < 10:
        return np.full(num_assets, 1 / num_assets)
    mean_returns = returns.mean()
    cov_matrix = make_covariance_psd(returns.cov())
    w = cp.Variable(num_assets)
    risk = cp.quad_form(w, cov_matrix)
    problem = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
    problem.solve()
    weights = np.array(w.value) if w.value is not None else np.full(num_assets, 1 / num_assets)
    return weights / np.sum(weights)

def simulate(strategy, start_date=START_DATE, end_date=END_DATE):
    """Simulates a weekly investment strategy based on the selected method."""
    prices = fetch_data()
    prices = prices.loc[start_date:end_date]
    returns = compute_returns(prices)
    portfolio = {etf: 0 for etf in ETF_LIST}
    investment_log = []

    for date, row in prices.iterrows():
        if date not in returns.index:
            continue

        if strategy == "MVO":
            returns_so_far = returns.loc[:date].dropna()
            optimal_weights = mean_variance_optimization(returns_so_far)
        elif strategy == "SPY":
            optimal_weights = np.array([1.0 if etf == "SPY" else 0.0 for etf in ETF_LIST])
        else:
            raise ValueError("Invalid strategy. Choose 'MVO' or 'SPY'.")

        raw_allocations = WEEKLY_BUDGET * optimal_weights
        rounded_allocations = np.round(raw_allocations, 2)
        difference = WEEKLY_BUDGET - np.sum(rounded_allocations)
        if difference != 0:
            max_index = np.argmax(raw_allocations)
            rounded_allocations[max_index] += difference

        for i, etf in enumerate(ETF_LIST):
            price = row[etf]
            if not np.isnan(price) and rounded_allocations[i] > 0:
                shares = round(rounded_allocations[i] / price, 4)
                portfolio[etf] += shares
                investment_log.append([date.strftime('%Y-%m-%d'), etf, price, shares, rounded_allocations[i]])

    df = pd.DataFrame(investment_log, columns=["Date", "ETF", "Price", "Shares", "Total_Invested"])
    file_name = "simulation_results.csv" if strategy == "MVO" else "spy_dca_simulation.csv"
    df.to_csv(file_name, index=False)
    return df, portfolio

def plot_results(df, portfolio, strategy):
    """Plots the total portfolio value over time."""
    prices = fetch_data()
    portfolio_values = []
    for date, row in prices.iterrows():
        value = sum(portfolio.get(etf, 0) * row[etf] for etf in ETF_LIST if etf in portfolio and not pd.isna(row[etf]))
        portfolio_values.append([date, value])
    
    df_value = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio_Value"])
    df_value["Date"] = pd.to_datetime(df_value["Date"])
    df_value = df_value.dropna()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_value, x="Date", y="Portfolio_Value", label=f"Portfolio Value ({strategy})")
    plt.xlabel("Date")
    plt.ylabel("Value in USD")
    plt.title(f"Portfolio Value Over Time ({strategy})")
    plt.legend()
    plt.grid()
    plt.show()

def plot_portfolio_pie(portfolio):
    """Plots the portfolio allocation as a pie chart based on total investment value."""
    prices = fetch_data()
    latest_prices = prices.iloc[-1] 

    portfolio_values = {etf: portfolio[etf] * latest_prices[etf] for etf in portfolio if not pd.isna(latest_prices[etf])}
    total_value = sum(portfolio_values.values())

    plt.figure(figsize=(8, 8))
    plt.pie(portfolio_values.values(), labels=portfolio_values.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Allocation by Value")
    plt.show()


df_mvo, portfolio_mvo = simulate("MVO")
plot_results(df_mvo, portfolio_mvo, "MVO")
plot_portfolio_pie(portfolio_mvo)

df_spy, portfolio_spy = simulate("SPY")
plot_results(df_spy, portfolio_spy, "SPY")
plot_portfolio_pie(portfolio_spy)