# MVO+CDA Simulation tool

## Overview
This project implements an **MVO (Mean-Variance Optimization) strategy** for ETF investments using **Dollar-Cost Averaging (DCA)**. The application allows users to simulate their portfolio growth over time( using past data ) and visualize asset allocation using **Streamlit**.

![image](https://github.com/user-attachments/assets/3a79dcda-1b2f-4ea0-8b84-bdab35105a89)

## Features
- **MVO Investment Strategy**: Dynamically allocates weekly investments based on risk-adjusted returns.
- **Streamlit Dashboard**: Interactive UI to customize the simulation.
- **Custom Weekly Budget**: Users can set their preferred weekly investment amount.
- **Portfolio Growth Visualization**: Compares portfolio performance with the S&P 500.
- **Asset Allocation Pie Chart**: Shows ETF distribution based on current market value.
- **Simulation Logging**: Stores investment history in CSV format for analysis.

## File Structure
```
DCA-ETF-Autotrader
 ┣ app.py                 # Streamlit application
 ┣ invest.py              # Investment strategy implementation
 ┣ etf_data.csv           # Stored ETF historical data
 ┣ requirements.txt        # Python dependencies
 ┣ README.md              # Project documentation
```

## Installation
Clone the repository and install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the Streamlit application:
```sh
streamlit run app.py
```
Then, open the browser to explore the interactive investment simulator.

## Example Simulation Results

Portfolio Value Over time
![image](https://github.com/user-attachments/assets/d269edb4-c278-40da-b174-1c7b0d822122)

Investments distribution
![image](https://github.com/user-attachments/assets/1090cb05-b5ee-49eb-a87c-7c6df870020d)

## Built With
- **Python** (Pandas, NumPy, CVXPY)
- **Streamlit** (for UI)
- **Matplotlib** (for visualization)
