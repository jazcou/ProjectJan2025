#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:15:55 2025

@author: jasmincoughtrey
"""
import pandas as pd
import os
import gc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
start_time="02:00:00"
end_time="08:00:00"
#Define month codes and expiration days (assuming CME contract rules)



#ES, MES, ZN, etc
month_codes = {
    'M': [4,5,6],    # January
    'U': [7,8,9],    # February
    'Z': [10,11,12],    # March
    'H': [1,2,3]}    # April
 


contract_expiration_days = {
    'CL': 20,   # Crude Oil expires on the 25th of the month
    'ES': 21,   # E-mini S&P 500 Futures
    'GC': 26,   # Gold Futures
    'HG': 26,   # Copper Futures
    'MES': 21,  # Micro E-mini S&P 500 Index Futures
    'MNH': 16,  # Micro USD/CNH Futures
    'MNQ': 21,  # Micro E-mini Nasdaq-100 Index Futures
    'ZF': 21,   # 5-Year T-Note Futures
    'ZN': 21,   # 10-Year T-Note Futures
    'ZT': 21    # 2-Year T-Note Futures
}


# Parameters
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
chunk_size = 1000000  # Define chunk size for large files
file_path = r"/Users/jasmincoughtrey/Downloads/data_project.csv"
symbology_csv = r"/Users/jasmincoughtrey/Downloads/symbology.csv"
symbology = pd.read_csv(symbology_csv)
symbology['date'] = pd.to_datetime(symbology['date'])
#symbology = symbology[symbology['raw_symbol'].str.match(f'^{asset}[A-Z]\\d$')]
#symbology.sort_values(by='max')
# Calculate start and end dates for each raw_symbol (contract)
#symbology = symbology.groupby('raw_symbol')['date'].agg(['min', 'max']).reset_index()
#assets = ["CL", "ES", "GC", "HG", "MES", "MNH", "MNQ", "ZF", "ZN", "ZT"]
assets = ["CL"]  # Start with CL (Crude Oil); add more assets as needed

from calendar import monthrange


def get_month_code(month):
    """
    Retrieve the month code for a given month number.
    """
    for code, months in month_codes.items():
        # Check if the month is in the list or directly equals the value
        if isinstance(months, list) and month in months:
            return code
        elif month == months:  # Handle single-value cases
            return code
    raise ValueError(f"No month code found for month: {month}")
    
    
def generate_contract_dates(asset, start_date, end_date, roll_days=5):
    """
    Generate contracts with dynamic handling of days in a month and ensure no overlaps.
    Contracts will start at `00:00:00` and end at `23:59:59` on the calculated roll day.
    """
    contracts = []
    current_date = start_date

    # Adjust the first contract to start on the `min` date at `00:00:00`
    prev_end_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

    while current_date <= end_date:
        # Get month and year
        month = current_date.month
        year = current_date.year

        # Get month code
        month_code = get_month_code(month)
        #month_code = [code for code, m in month_codes.items() if m == month][0]

        # Determine the last day of the month dynamically
        last_day = monthrange(year, month)[1]  # Get the number of days in the current month
        expiry_date = datetime(year, month, min(last_day, contract_expiration_days[asset]), 23, 59, 59)
        roll_date = expiry_date - timedelta(days=roll_days)

        # Set start and end dates for the contract
        start_contract_date = prev_end_date + timedelta(days=1)
        start_contract_date = start_contract_date.replace(hour=0, minute=0, second=0, microsecond=0)  # Start at 00:00:00
        end_contract_date = roll_date.replace(hour=23, minute=59, second=59, microsecond=0)  # End at 23:59:59

        # Append the contract details
        if month == 12:
            contracts.append({
                'raw_symbol': f"{asset}{month_code}{str(year+1)[-1]}",
                'start_date': start_contract_date,
                'end_date': end_contract_date
            })            
        else:            
            contracts.append({
                'raw_symbol': f"{asset}{month_code}{str(year)[-1]}",
                'start_date': start_contract_date,
                'end_date': end_contract_date
            })

        # Update prev_end_date to the end of the current contract
        prev_end_date = end_contract_date

        # Move to the next month
        if month == 12:  # Handle year-end
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year

        # Move to the first day of the next month
        current_date = datetime(next_year, next_month, 1, 0, 0, 0)  # Ensure the time is 00:00:00

    return pd.DataFrame(contracts)


# Modified processing to include contract-based filtering
for asset in assets:
    print(f"Processing contracts for asset: {asset}")

    # Generate contracts for the asset
    contracts = generate_contract_dates(asset, datetime(2022, 1, 1, 0, 0, 0), datetime(2024, 12, 31, 23, 59, 59))

    print(f"Generated contracts for {asset}:")
    print(contracts)

    # Process the data file in chunks
    filtered_chunks = []
    chunk_counter = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_counter += 1
        print(f"Processing chunk {chunk_counter} for {asset}...")

        # Convert timestamp and filter rows
        chunk['ts_recv'] = pd.to_datetime(chunk['ts_recv'].str.strip(), format='%Y-%m-%dT%H:%M:%S.%fZ')
        filtered_chunk = chunk[chunk['symbol'].str.match(rf'^{asset}[FGHJKMNQUVXZ]\d$')]       
        filtered_chunks.append(filtered_chunk)

        # if chunk_counter == 65:
        #     break

    # Combine all filtered chunks into a single DataFrame
    if filtered_chunks:
        df = pd.concat(filtered_chunks).sort_values(by='ts_recv').reset_index(drop=True)
        print(f"Data for {asset} successfully filtered. Shape: {df.shape}")

        df = df.dropna()
        # Select the desired columns
        new_df = df[['ts_recv', 'price', 'bid_px_00', 'ask_px_00', 'instrument_id', 'symbol']]

        # Rename columns if needed (optional)
        new_df.rename(columns={'bid_px_00': 'bid', 'ask_px_00': 'ask', 'ts_recv': 'datetime'}, inplace=True)

        df = new_df
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        combined_df = pd.DataFrame()
        for _, contract in contracts.iterrows():
            filtered_df = df[
                (df['symbol'] == contract['raw_symbol']) &
                (df['datetime'] >= contract['start_date']) &
                (df['datetime'] <= contract['end_date'])]
            combined_df = pd.concat([combined_df, filtered_df], ignore_index=True)

        # Sort by datetime for proper ordering
        combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)
        


        df = combined_df
        

        # Filter within the desired time period
        df = df[(df['datetime'].dt.time >= pd.to_datetime(start_time).time()) &
                (df['datetime'].dt.time < pd.to_datetime(end_time).time())]

        df['price_diff'] = df['price'].diff()
        df['adj_price'] = df['price_diff'].cumsum()

        # Save the filtered data
        df.to_parquet(f"{asset}_filtered_data.parquet", index=False)
        # Ensure your datetime column is in datetime format
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Plot the time series
        plt.figure(figsize=(12, 6))  # Set the figure size
        plt.plot(df['datetime'], df['adj_price'], label='Adjusted Price')
        
        # Add labels and title
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('Adjusted Price', fontsize=12)
        plt.title('Adjusted Price Time Series', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Show the plot
        plt.tight_layout()  # Ensure everything fits nicely
        plt.show()
        






