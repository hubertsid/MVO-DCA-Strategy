import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from invest import simulate, fetch_data

st.title("📊 Investment Simulation App")
st.sidebar.header("⚙️ Simulation Settings")

years_back = st.sidebar.slider("Select Simulation Period (Years)", 1, 20, 5)
start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

weekly_budget = st.sidebar.number_input("Set Weekly Investment Budget ($):", min_value=10, max_value=500, value=20)

def plot_portfolio_pie(portfolio):
    """Plots the portfolio allocation as a pie chart based on total investment value."""
    prices = fetch_data()
    latest_prices = prices.iloc[-1] 
    portfolio_values = {etf: portfolio[etf] * latest_prices[etf] for etf in portfolio if not pd.isna(latest_prices[etf])}
    total_value = sum(portfolio_values.values())
    
    if total_value == 0:
        st.write("⚠️ No investments made yet. Cannot create pie chart.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(portfolio_values.values(), labels=portfolio_values.keys(), autopct='%1.1f%%', startangle=140)
    ax.set_title("Portfolio Allocation by Value")
    st.pyplot(fig)

if st.sidebar.button("🚀 Run Simulation"):
    st.write(f"Running MVO strategy from {start_date} to {end_date} with a weekly budget of ${weekly_budget}...")
    df, portfolio = simulate("MVO", start_date=start_date, end_date=end_date, weekly_budget=weekly_budget)
    st.success("✅ Simulation Completed!")
    
    st.subheader("📊 Portfolio Allocation")
    plot_portfolio_pie(portfolio)

    st.subheader("📋 Investment Log")
    st.dataframe(df)