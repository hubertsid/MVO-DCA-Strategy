import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

ETF_LIST = [
    "QQQ", "TQQQ", "SPY", "VOO", "VTI", "DIA", "IAU",
    "EEM", "VWO", "XLK", "XLV", "XLF", "XLY", "XLE", "SMH"
]

START_DATE = "2010-09-09"
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

def simulate(strategy, etf_list=ETF_LIST, start_date=START_DATE, end_date=END_DATE, weekly_budget=20):
    """Simulates a weekly investment strategy based on the selected method."""
    prices = fetch_data()
    prices = prices.loc[start_date:end_date, etf_list]
    returns = compute_returns(prices)
    portfolio = {etf: 0 for etf in etf_list}
    investment_log = []

    for date, row in prices.iterrows():
        if date not in returns.index or date.weekday() != 0:
            continue

        if strategy == "MVO":
            returns_so_far = returns.loc[:date].dropna()
            valid_columns = returns_so_far.columns[~returns_so_far.iloc[-1].isna()]
            if len(valid_columns) < 2:
                continue

            filtered_returns = returns_so_far[valid_columns]
            optimal_weights_sub = mean_variance_optimization(filtered_returns)

            optimal_weights = np.zeros(len(etf_list))
            for i, etf in enumerate(etf_list):
                if etf in valid_columns:
                    idx = list(valid_columns).index(etf)
                    optimal_weights[i] = optimal_weights_sub[idx]

        elif strategy == "SPY":
            optimal_weights = np.array([1.0 if etf == "SPY" else 0.0 for etf in etf_list])
        else:
            raise ValueError("Invalid strategy. Choose 'MVO' or 'SPY'.")

        raw_allocations = weekly_budget * optimal_weights
        rounded_allocations = np.round(raw_allocations, 2)
        difference = weekly_budget - np.sum(rounded_allocations)
        if difference != 0:
            max_index = np.argmax(raw_allocations)
            rounded_allocations[max_index] += difference

        for i, etf in enumerate(etf_list):
            price = row[etf]
            if not np.isnan(price) and rounded_allocations[i] > 0:
                shares = round(rounded_allocations[i] / price, 4)
                portfolio[etf] += shares
                investment_log.append([
                    date.strftime('%Y-%m-%d'), etf, price, shares, rounded_allocations[i]
                ])

    df = pd.DataFrame(investment_log, columns=["Date", "ETF", "Price", "Shares", "Total_Invested"])
    file_name = "simulation_results.csv" if strategy == "MVO" else "spy_dca_simulation.csv"
    df.to_csv(file_name, index=False)
    return df, portfolio

def build_portfolio_history(investment_df, etf_list=ETF_LIST):
    """Returns history of portfolio values"""
    investment_df["Date"] = pd.to_datetime(investment_df["Date"])
    investment_df = investment_df.sort_values("Date")
    
    dates = investment_df["Date"].unique()
    portfolio_history = []
    portfolio = {etf: 0 for etf in etf_list}

    for date in dates:
        day_data = investment_df[investment_df["Date"] == date]
        for _, row in day_data.iterrows():
            portfolio[row["ETF"]] += row["Shares"]
        snapshot = {"Date": date}
        snapshot.update(portfolio.copy())
        portfolio_history.append(snapshot)

    return pd.DataFrame(portfolio_history)

def compute_portfolio_value(portfolio_history_df, prices_df, etf_list=ETF_LIST):
    """Calculates final value"""
    prices_df = prices_df.copy()
    prices_df.index = pd.to_datetime(prices_df.index)

    etf_list = [etf for etf in etf_list if etf in prices_df.columns]

    history = portfolio_history_df.copy()
    history["Date"] = pd.to_datetime(history["Date"])
    history = history.set_index("Date")
    common_dates = history.index.intersection(prices_df.index)

    history = history.loc[common_dates]
    prices_df = prices_df.loc[common_dates, etf_list]

    portfolio_values = (history[etf_list] * prices_df[etf_list]).sum(axis=1).reset_index()
    portfolio_values.columns = ["Date", "Portfolio_Value"]
    return portfolio_values


def plot_results(portfolio_values_df, strategy):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=portfolio_values_df, x="Date", y="Portfolio_Value", label=f"Portfolio Value ({strategy})", ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value in USD")
    ax.set_title(f"Portfolio Value Over Time ({strategy})")
    ax.grid(True)
    ax.legend()
    return fig

def plot_portfolio_pie(portfolio, etf_list=ETF_LIST):
    prices = fetch_data()
    latest_prices = prices.iloc[-1] 

    portfolio_values = {
        etf: portfolio[etf] * latest_prices[etf]
        for etf in etf_list
        if etf in portfolio and not pd.isna(latest_prices[etf])
    }

    total_value = sum(portfolio_values.values())
    if total_value == 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(portfolio_values.values(), labels=portfolio_values.keys(), autopct='%1.1f%%', startangle=140)
    ax.set_title("Portfolio Allocation by Value")
    return fig

def compare_strategies_plot(value_mvo, value_spy, weekly_budget=20):
    start_date = min(value_mvo["Date"].min(), value_spy["Date"].min())
    end_date = max(value_mvo["Date"].max(), value_spy["Date"].max())
    date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')

    cash_df = pd.DataFrame({"Date": date_range})
    cash_df["Portfolio_Value"] = (cash_df.index + 1) * weekly_budget

    value_mvo = value_mvo.copy()
    value_mvo["Strategy"] = "MVO"

    value_spy = value_spy.copy()
    value_spy["Strategy"] = "S&P500 DCA"

    cash_df = cash_df.copy()
    cash_df["Strategy"] = "Cash Only"

    combined = pd.concat([value_mvo, value_spy, cash_df])

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=combined, x="Date", y="Portfolio_Value", hue="Strategy", ax=ax)
    ax.set_title("Portfolio Value Comparison Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.grid(True)
    ax.legend(title="Strategy")
    return fig
