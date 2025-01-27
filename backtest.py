#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:55:49 2025

@author: jasmincoughtrey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# need to create backtest capable of forecasting any horizon to switch to daily 
def backtest_multiple_portfolios_with_vol_adjustment_and_costs(results, data, assets, target_vol=0.01, vol_window=252):

    # initialise strategy P&Ls
    equally_weighted_pnl = []
    risk_parity_pnl = []
    conviction_weighted_pnl = []
    price_combined_pnl = []

    # Rolling volatilities for risk parity
    rolling_vols = data[[f"{asset}_percent_returns" for asset in assets]].rolling(vol_window).std()

    # Transaction costs: Assume bid-ask spread as a proxy for costs
    transaction_costs = {asset: data[f"{asset}_bid_ask_bps"] / 10000 for asset in assets}

    # Initialise portfolio weights
    prev_weights = {strategy: np.zeros(len(assets)) for strategy in ['Equally Weighted', 'Risk Parity', 'Conviction Weighted']}

    # Iterate through results
    for i, (_, res) in enumerate(results.items()):
        predictions = {asset: res[f"{asset}_Meta_Model_Out_Sample_Predictions"] for asset in assets}
        targets = {asset: res[f"{asset}_Out_Sample_Target"] for asset in assets}

        # Conviction weights (absolute values of predictions)
        

        # Equally weighted portfolio

        # Risk parity portfolio


        # Conviction-weighted portfolio

        # Combined asset prices (simple price-based strategy)

        # Compute transaction costs for each strategy


        # Adjust returns for costs


        # Append to P&L lists
    

        # Update previous weights

    # Combine all P&Ls
    strategies = {

    # Volatility Adjust Each Strategy
 
    # Plot results

    return strategies, sharpe_ratios




#### dynamically update?
# work out amount to be traded 
# costs
# contract size etc bring that in to work out amount to be traded 
# workout based on minute bars forecast horizon to add up all thats being traded and position sizes to set it to daily
# bring back stats in pdf on diversification
# drawdown and holdning pposition