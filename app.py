import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from invest import (
    simulate, fetch_data,
    build_portfolio_history, compute_portfolio_value,
    plot_results, plot_portfolio_pie, compare_strategies_plot
)

st.set_page_config(layout="wide")
st.title("Investment Simulation App")
st.sidebar.header("Simulation Settings")

years_back = st.sidebar.slider("Select Simulation Period (Years)", 1, 15, 5)
start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

weekly_budget = st.sidebar.number_input("Set Weekly Investment Budget ($):", min_value=10, max_value=500, value=20)

prices = fetch_data()
etf_options = list(prices.columns)

DEFAULT_ETF_LIST = [
    "VTI", "VOO", "QQQ", "IAU", "VWO", "TQQQ", "SPY"
]


selected_etfs = st.sidebar.multiselect(
    "Choose ETF(s) to include in the strategy:",
    options=etf_options,
    default=[etf for etf in DEFAULT_ETF_LIST if etf in etf_options]
)

if st.sidebar.button("Run Simulation"):
    if len(selected_etfs) < 2:
        st.warning("Please select at least two ETF options.")
    else:
        st.write(f"Running MVO strategy from {start_date} to {end_date} with a weekly budget of ${weekly_budget}...")

        df_mvo, portfolio_mvo = simulate(
            "MVO",
            etf_list=selected_etfs,
            start_date=start_date,
            end_date=end_date,
            weekly_budget=weekly_budget
        )
        history_mvo = build_portfolio_history(df_mvo, etf_list=selected_etfs)
        value_mvo = compute_portfolio_value(history_mvo, prices, etf_list=selected_etfs)

        df_spy, portfolio_spy = simulate(
            "SPY",
            etf_list=selected_etfs,
            start_date=start_date,
            end_date=end_date,
            weekly_budget=weekly_budget
        )
        history_spy = build_portfolio_history(df_spy, etf_list=selected_etfs)
        value_spy = compute_portfolio_value(history_spy, prices, etf_list=selected_etfs)

        st.subheader("Portfolio Value Over Time (MVO)")
        fig_compare = compare_strategies_plot(value_mvo, value_spy, weekly_budget=weekly_budget)
        st.pyplot(fig_compare)

        st.subheader("Final Portfolio Value")
        final_value = value_mvo["Portfolio_Value"].iloc[-1]
        st.metric(label="Total Value", value=f"${final_value:,.2f}")

        st.subheader("Portfolio Allocation")
        fig_pie = plot_portfolio_pie(portfolio_mvo, etf_list=selected_etfs)
        if fig_pie:
            st.pyplot(fig_pie)
        else:
            st.write("No data available for pie chart.")

        st.subheader("Investment Log")
        st.dataframe(df_mvo)