#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 00:07:16 2025

@author: jasmincoughtrey
"""
import pandas as pd
import os
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

assets = ["ES", "HG", "MES", "MNQ", "ZF", "ZN", "ZT", "GC"]
#assets = ["CL","ES"]
dfs = []



# List to store processed DataFrames
dfs = []

for asset in assets:
    # Read the Parquet file
    df = pd.read_parquet(f"{asset}_filtered_data.parquet")
    
    # Ensure your datetime column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Plot the time series
    plt.figure(figsize=(12, 6))  # Set the figure size
    plt.plot(df['datetime'], df['adj_price'], label='Adjusted Price')
    
    # Add labels and title
    plt.xlabel('Date and Time', fontsize=12)
    plt.ylabel(f'Adjusted Price {asset}', fontsize=12)
    plt.title(f'Adjusted Price Time Series for {asset}', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()  # Ensure everything fits nicely
    plt.show()
    
    

    # Set 'datetime' as index
    df.set_index('datetime', inplace=True)
    
    # Rename columns to include asset prefix
    df.rename(columns=lambda col: f"{asset}_{col}" if col != 'datetime' else col, inplace=True)
    
    
    # Append to the list of DataFrames
    dfs.append(df)



merged_df = pd.concat(dfs, axis=1)
merged_df.ffill(inplace=True)
merged_df = merged_df.dropna()
print(merged_df)
merged_df.to_parquet("merged_assets.parquet")