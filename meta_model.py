#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:26:12 2025

@author: jasmincoughtrey
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import torch
from scipy.fftpack import fft
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import f_regression
import seaborn as sns
import shap
from pykalman import KalmanFilter
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import accuracy_score


# Define autoencoder architecture
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the function to generate noise
def generate_noise_reduction_features(data, asset, lookback_windows,autoencoders_='true'):

    features = pd.DataFrame(index=data.index)
    price_col = f"{asset}_adj_price"
    percent_returns_col = f"{asset}_percent_returns"
    features['price'] = data[price_col]
    features['percent_returns'] = data[percent_returns_col]
    try:
        # apply kalman -filter set to nan if error
        kf_prices = KalmanFilter(
            initial_state_mean=features['price'].iloc[0],
            initial_state_covariance=1.0, 
            transition_matrices=[1], 
            observation_matrices=[1], 
            observation_covariance=1.0,
            transition_covariance=0.01
        )
        state_means_prices, _ = kf_prices.filter(features['price'].fillna(0).values)
        features["Kalman_Filtered_Price"] = state_means_prices
    except Exception as e:
        print(f"Kalman Filter failed on raw prices: {e}")
        features["Kalman_Filtered_Price"] = np.nan
        # run smoothed features over varying windows
    for window in lookback_windows:
        # Apply Fourier Transform 
        rolling_price_mean = features['price'].rolling(window=window).mean().fillna(0)
        fft_result = np.abs(fft(rolling_price_mean.to_numpy()))  # Convert to NumPy array
        features[f"Fourier_Magnitude_{window}"] = fft_result[:len(features)]
        # if autoencoder set to true run
        if autoencoders_ == 'true':  
            print('...generating autoencoder features')
            # Smoothed Returns for Autoencoder
            rolling_returns = features['percent_returns'].rolling(window=window).mean().fillna(0)
            if len(rolling_returns) > 0:
                ae_model = Autoencoder(input_dim=1, hidden_dim=5)
                optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.01)
                loss_fn = torch.nn.MSELoss()
    
                x_tensor = torch.tensor(rolling_returns.values, dtype=torch.float32).unsqueeze(-1)
                for epoch in range(50):  # Train Autoencoder
                    optimizer.zero_grad()
                    output = ae_model(x_tensor)
                    loss = loss_fn(output, x_tensor)
                    loss.backward()
                    optimizer.step()
                features[f"Smoothed_Returns_{window}"] = output.detach().numpy().flatten()
            
        try:
            # also apply kalman on returns to complement the use case, focusing more on volatility regimes, short-term dynamics, and noise-reduced signals for trading strategies
            rolling_returns = features['percent_returns'].rolling(window=window).mean().fillna(0)
            kf_returns = KalmanFilter(
                initial_state_mean=rolling_returns.iloc[0],
                initial_state_covariance=1.0,
                transition_matrices=[1],
                observation_matrices=[1], 
                observation_covariance=1.0,
                transition_covariance=0.01
            )
            state_means_returns, _ = kf_returns.filter(rolling_returns.values)
            features[f"Kalman_Filtered_Returns_{window}"] = state_means_returns
        except Exception as e:
            print(f"Kalman Filter failed for window {window}: {e}")
            features[f"Kalman_Filtered_Returns_{window}"] = np.nan

    return features





def generate_trend_features(data, asset, lookback_windows):
    # generate relatively simple trend feature to capture long term patterns
    features = pd.DataFrame(index=data.index)
    price_col = f"{asset}_adj_price"
    percent_returns_col = f"{asset}_percent_returns"
    features['price'] = data[price_col]
    features['percent_returns'] = data[percent_returns_col]
    for window in lookback_windows:
        features[f"SMA_{window}"] = features['price'].rolling(window=window).mean()
        features[f"EMA_{window}"] = features['price'].ewm(span=window, adjust=False).mean()
        features[f"Momentum_{window}"] = features['price'].diff(window)
        features[f"ROC_{window}"] = (features['price'].diff(window) / features['price'].shift(window)) * 100
        features[f"Volatility_{window}"] = features['percent_returns'].rolling(window=window).std()
        delta = features['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        features[f"RSI_{window}"] = 100 - (100 / (1 + rs))
        rolling_mean = features['percent_returns'].rolling(window=window).mean()
        rolling_std = features['percent_returns'].rolling(window=window).std()
        features[f"Sharpe_{window}"] = (rolling_mean / rolling_std)* np.sqrt(window)
        downside = features['percent_returns'].rolling(window=window).apply(lambda x: np.sqrt(np.mean(np.square(x[x < 0]))), raw=False)
        features[f"Sortino_{window}"] = (rolling_mean / downside  )* np.sqrt(window)
        signal = features['price'].rolling(window=window).mean()
        noise = features['price'].rolling(window=window).std()
        features[f"SNR_{window}"] = signal / noise
        features["MACD"] = features['price'].ewm(span=window, adjust=False).mean() - features['price'].ewm(span=window*2, adjust=False).mean()
        features["Signal_Line"] = features["MACD"].ewm(span=window, adjust=False).mean()
        features[f"MACD_Histogram_{window}"] = features["MACD"] - features["Signal_Line"]

  #  Drop rows with NaN values caused by rolling operations to not feed in to model
    features = features.dropna()

    return features





def generate_revserion_features(data, asset, lookback_windows, gnn_features= False):
    features = pd.DataFrame(index=data.index)
    price_col = f"{asset}_adj_price"
    percent_returns_col = f"{asset}_percent_returns"
    bid_ask_diff_col = f"{asset}_bid_ask_bps"
    
    # vol_percent_change_col = f"{asset}_vol_percent_change"
    features['price'] = data[price_col]
    features['percent_returns'] = data[percent_returns_col]
    features['bid_ask_diff'] = ((data[f'{asset}_ask'] - data[f'{asset}_bid']) /
                                               data[f'{asset}_adj_price'] * 10000)
    features['Bid'] = data[f"{asset}_bid"]
    features['Ask'] = data[f"{asset}_ask"]
    features['Mid'] = data[price_col]
    # features['vpc'] = data[vol_percent_change_col]
    print('..core complete')
    for window in lookback_windows:
        rolling_mean = features['percent_returns'].rolling(window=window).mean()
        rolling_mean_price = features['price'].rolling(window=window).mean()
        rolling_std = features['percent_returns'].rolling(window=window).std()
        rolling_max = features['price'].rolling(window=window).max()
        rolling_min = features['price'].rolling(window=window).min()
        features[f"Distance_Above_Upper_Bollinger_{window}"] = np.where(features['price'] > (rolling_mean + (2 * rolling_std)),features['percent_returns'],0)
        features[f"Distance_below_lower_Bollinger_{window}"] = np.where(features['price'] < (rolling_mean - (2 * rolling_std)),features['percent_returns'],0)
        features[f"Z_Score_{window}"] = (features['percent_returns'] - rolling_mean) / rolling_std
        features[f"Drawdown_{window}"] = (rolling_max - features['price']) / rolling_max
        features[f"Price_Spike_{window}"] = np.abs(features['price'] - rolling_mean_price) / rolling_mean_price

        # microstructure
        features[f"Bid_Ask_Expansion_{window}"] = features['bid_ask_diff'] / features['bid_ask_diff'].rolling(window=window).mean()
        features[f"Ask_Diff_{window}"] = (features['Ask'].rolling(window=window).mean()-features['Mid'].rolling(window=window).mean())/features['Mid'].rolling(window=window).mean()
        features[f"Bid_Diff_{window}"] = (features['Mid'].rolling(window=window).mean()-features['Bid'].rolling(window=window).mean())/features['Mid'].rolling(window=window).mean()
        
        # # vol features 
        # features[f"Volatility_Spike_{window}"] = features['vpc']/features['vpc'].rolling(window=window).mean() 
        # print(f"..loop {window}")
        
        #correlation
        for other_asset in [f"{asset}" for i in range(1, 11) if f"asset{i}" != asset]:
            other_col = f"{other_asset}_percent_returns"
            # Calculate rolling correlation between target and other asset
            feature_name = f"{asset}_Corr_with_{other_asset}_{window}"
            features[feature_name] = features['percent_returns'].rolling(window=window).corr(data[other_col])



        # Trend Reversal Features
        features[f"Directional_Change_{window}"] = features['price'].diff().rolling(window=window).apply(
            lambda x: (np.sign(x) != np.sign(x.shift(1))).sum(), raw=False)
        features[f"Momentum_Flip_{window}"] = features['price'].diff(window).apply(np.sign).diff().abs()
        features[f"Price_vs_Max_{window}"] = (features['price'] - rolling_max)
        features[f"Price_vs_Min_{window}"] = (features['price'] - rolling_min)
        features[f"Relative_Drawdown_{window}"] = features['price'] / rolling_max - 1
        features["MACD"] = features['price'].ewm(span=window, adjust=False).mean() - features['price'].ewm(span=window*2, adjust=False).mean()
        features["Signal_Line"] = features["MACD"].ewm(span=window, adjust=False).mean()
        features[f"MACD_Histogram_{window}"] = features["MACD"] - features["Signal_Line"]     

    # Include GNN Features (can ignore)- now not using for this algortihm but was set up for original framework
    if gnn_features is True:
        gnn_features_aligned = gnn_features.loc[features.index]
        for col in gnn_features_aligned.columns:
            features[col] = gnn_features_aligned[col]

   # Drop rows with NaN values caused by rolling operations
    features = features.dropna()

    return features

def plot_hierarchical_clustering(features, corr_threshold=0.95):
    # bring backblot to visually see correlation and where the cut off will be
    corr_matrix = features.corr().abs()
    link = linkage(corr_matrix, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(link, labels=features.columns, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Features")
    plt.axhline(y=corr_threshold, color='r', linestyle='--', label=f"Threshold: {corr_threshold}")
    plt.legend()
    plt.show()

def select_features(features, target, selection="tree", n_components=3, remove_corr=True, corr_threshold=0.95):

    # Step 1: Apply feature selection method
    if selection == "tree":
        # Tree-based feature importance
        tree_model = DecisionTreeRegressor(max_depth=5)
        tree_model.fit(features, target)
        feature_importance = tree_model.feature_importances_

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(features.columns, feature_importance)
        plt.xticks(rotation=90)
        plt.title("Feature Importance from Tree-Based Model")
        plt.show()

        # Select top n_components based on importance
        top_features = np.argsort(feature_importance)[::-1][:n_components]
        return features.iloc[:, top_features], features.columns[top_features]

    elif selection == "f_test":
        # F-test for feature relevance
        f_values, p_values = f_regression(features, target)
        f_scores = pd.DataFrame({"Feature": features.columns, "F-Value": f_values, "P-Value": p_values})
        f_scores = f_scores.sort_values(by="F-Value", ascending=False)
        print(f"F-Test Results:\n{f_scores}")

        # Select top n_components based on F-scores
        top_features = f_scores.iloc[:n_components]["Feature"].values
        return features[top_features], top_features

    else:
        raise ValueError("Invalid selection for feature geenration")

    # Step 2: Potentially look to do a runn assesing the removal of highly correlated features
    if remove_corr:
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
        features = features.drop(columns=high_corr_features)

        plot_hierarchical_clustering(features, corr_threshold)

    return features, None

        
def generate_vol_features(data, asset, lookback_windows):

    features = pd.DataFrame(index=data.index)
    percent_returns_col = f"{asset}_percent_returns"
    # vol_col = f"asset{asset}_vol_percent_change"

    for window in lookback_windows:
        rolling_vol = data[percent_returns_col].rolling(window=window).std()
        relative_vol = rolling_vol / rolling_vol.mean()
        
        features[f"Rolling_Vol_{window}"] = rolling_vol
        features[f"Relative_Vol_{window}"] = relative_vol
        
        # features[f"Volatility_Spike_{window}"] = (
        #     data[vol_col].rolling(window=window).mean() / data[vol_col].rolling(window=window).std()
        # )
        features[f"Z_Score_Vol_{window}"] = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()
    
   # Drop rows with NaN values caused by rolling operations
    features = features.dropna()
    return features        

def reverse_engineer_weights(model, features):

    # a additional genrated interpritibility function to analyse network weights 
    try:
        # Check the number of layers in the model
        num_layers = len(model.coefs_)
  
        # adapt for number of layers
        if num_layers == 1:  
            weights_layer_1 = model.coefs_[0]  # Weights from input to hidden layer
            weights_output = model.coefs_[1]  # Weights from hidden layer to output

            biases_layer_1 = model.intercepts_[0]
            biases_output = model.intercepts_[1]

            # Forward pass through single layer
            activation_1 = np.maximum(0, np.dot(features, weights_layer_1) + biases_layer_1)
            output = np.dot(activation_1, weights_output) + biases_output  #

            # Reverse engineering weights of each
            layer_1_contribution = weights_output.T * activation_1  
            feature_importance = np.dot(layer_1_contribution.T, weights_layer_1.T).mean(axis=0)

        elif num_layers == 2: 
            weights_layer_1 = model.coefs_[0] 
            weights_layer_2 = model.coefs_[1]  
            weights_output = model.coefs_[2]  

            biases_layer_1 = model.intercepts_[0]
            biases_layer_2 = model.intercepts_[1]
            biases_output = model.intercepts_[2]

      
            activation_1 = np.maximum(0, np.dot(features, weights_layer_1) + biases_layer_1)  # ReLU for layers 
            activation_2 = np.maximum(0, np.dot(activation_1, weights_layer_2) + biases_layer_2)  
            output = np.dot(activation_2, weights_output) + biases_output  

            # Reverse engineering contribution
            layer_2_contribution = weights_output.T * activation_2  
            layer_1_contribution = np.dot(layer_2_contribution, weights_layer_2.T) * activation_1  
            feature_importance = np.dot(layer_1_contribution.T, weights_layer_1.T).mean(axis=0)

        else:
            raise ValueError(f"Inocrrect number of layers")

        # normalise importance values
        feature_importance = feature_importance / np.sum(np.abs(feature_importance))

        # bring back results
        importance_df = pd.DataFrame({
            'Feature': features.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        return importance_df

    except IndexError as e:
        print(f"IndexError {e}")

        return pd.DataFrame({'Feature': features.columns, 'Importance': [0] * features.shape[1]}) 

    except Exception as e:
        print(f"An error occurred in reverse_engineer_weights: {e}")
        return pd.DataFrame({'Feature': features.columns, 'Importance': [0] * features.shape[1]})  

def apply_pca(features_train, features_out_sample, n_components=3):

    pca = PCA(n_components=n_components)
    train_pca_features = pd.DataFrame(
        pca.fit_transform(features_train),
        index=features_train.index,
        columns=[f"PCA_{i+1}" for i in range(n_components)]
    )
    out_sample_pca_features = pd.DataFrame(
        pca.transform(features_out_sample),
        index=features_out_sample.index,
        columns=[f"PCA_{i+1}" for i in range(n_components)]
    )
    explained_variance = pca.explained_variance_ratio_
    
    return train_pca_features, out_sample_pca_features

def fit_model(train_features, train_target, test_features, test_target, model_type):

    # split data into train window
    X_train = train_features
    y_train = train_target

    # split data into test window
    X_test = test_features
    y_test = test_target

    # define ranges for netowrk and in turn heat map
    hidden_layer_range = range(5, 25,5)  
    epoch_range = range(100, 500, 100) 

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'tree':
        model = DecisionTreeRegressor(
        max_depth=5,             # tree depth 
        min_samples_split=10,    # minimum samples required to split 
        min_samples_leaf=5,      # minimum samples required in a leaf node
        max_features="sqrt",     # subset of features at each split
        random_state=42  )       # seed for reproducibility
    elif model_type == 'boosted_tree':
        model = GradientBoostingRegressor(
        n_estimators=100,        # no. of boosting stages
        learning_rate=0.1,       
        max_depth=3,             
        min_samples_split=10,    
        min_samples_leaf=5,      
        max_features="sqrt",    
        subsample=0.8,           # Fsamples used for fitting individual trees
        random_state=42          
    )
    elif model_type in ['nn_1_layer', 'nn_2_layer']:
        results_heatmap = []
        best_r2 = float('-inf')
        best_params = None

        # Loop through neuron and epoch combinations for heatmap
        for neurons in hidden_layer_range:
            row_results = []
            for epochs in epoch_range:
                # Set hidden layers
                if model_type == 'nn_1_layer':
                    model = MLPRegressor(
                        hidden_layer_sizes=(neurons,), 
                        max_iter=epochs, 
                        alpha=0.001,  # L2 regularisation to reduce overfitting
                        random_state=42
                    )
                elif model_type == 'nn_2_layer':
                    model = MLPRegressor(
                        hidden_layer_sizes=(neurons, neurons), 
                        max_iter=epochs, 
                        alpha=0.001,  # L2 regularisation to reduce overfitting
                        random_state=42
                    )
                try:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_train)
                    r2 = r2_score(y_train, predictions)
                    row_results.append(r2)

                    # keep a note of best paramas
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = (neurons, epochs)
                except Exception as e:
                    row_results.append(None)  
            
            results_heatmap.append(row_results)

        # Visualise the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(results_heatmap, annot=True, fmt=".2f", xticklabels=epoch_range, yticklabels=hidden_layer_range)
        plt.xlabel("Epochs")
        plt.ylabel("Neurons in Hidden Layer")
        plt.title(f"R2 Score Heatmap for {model_type}")
        plt.show()

        # Train the best model with optimal parameters
        print(f"{model_type}: Neurons={best_params[0]}, Epochs={best_params[1]} potimum params")
        model = MLPRegressor(hidden_layer_sizes=(best_params[0],) if model_type == 'nn_1_layer' else (best_params[0], best_params[0]),
                              max_iter=best_params[1], random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # final model on in-sample data
    model.fit(X_train, y_train)


    in_sample_predictions = model.predict(X_train)
    out_of_sample_predictions = model.predict(X_test)

    in_sample_pred_binary = (in_sample_predictions > 0).astype(int)
    out_of_sample_pred_binary = (out_of_sample_predictions > 0).astype(int)
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    

    in_sample_r2 = r2_score(y_train, in_sample_predictions)
    in_sample_mse = mean_squared_error(y_train, in_sample_predictions)
    in_sample_accuracy = accuracy_score(y_train_binary, in_sample_pred_binary)

    out_of_sample_r2 = r2_score(y_test, out_of_sample_predictions)
    out_of_sample_mse = mean_squared_error(y_test, out_of_sample_predictions)
    out_of_sample_accuracy = accuracy_score(y_test_binary, out_of_sample_pred_binary)


            
    # SHAP Analysis for Out-of-Sample Data
    shap_results = None
    shap_out_sample_correlations = None
    if model_type in ['nn_1_layer', 'nn_2_layer']:
        try:
            print("Generating SHAP values for out-of-sample data...")
            explainer = shap.Explainer(lambda x: model.predict(x), X_test)
            shap_values = explainer(X_test)

            # Extract SHAP values and mean SHAP values
            shap_values_matrix = shap_values.values
            mean_shap = np.abs(shap_values_matrix).mean(axis=0)

            # Compute correlations or R² with the out-of-sample target
            print("Computing SHAP-Target Correlations...")
            shap_out_sample_correlations = {}
            for i, feature in enumerate(X_test.columns):
                corr = np.corrcoef(shap_values_matrix[:, i], y_test)[0, 1]
                shap_out_sample_correlations[feature] = corr

            # Convert correlations to a DataFrame
            shap_out_sample_correlations = pd.DataFrame.from_dict(
                shap_out_sample_correlations, orient='index', columns=['Correlation_with_Target']
            ).sort_values(by='Correlation_with_Target', ascending=False)

            shap_results = {
                "shap_values": shap_values_matrix,
                "mean_shap": mean_shap,
                "out_sample_correlations": shap_out_sample_correlations
            }

            # SHAP summary plots
            shap.summary_plot(shap_values_matrix, X_test, plot_type="bar")
            shap.summary_plot(shap_values_matrix, X_test)

        except Exception as e:
            print(f"SHAP failed: {e}")
            
            

    if model_type in ['nn_1_layer', 'nn_2_layer']:
        print(f"Weights of {model_type}: {model.coefs_}")
        # Reverse engineer weights
        importance_df = reverse_engineer_weights(model, X_train)
    
     


    return {
        'Model': model,
        'Model': model,
        'In_Sample_R2': in_sample_r2,
        'In_Sample_MSE': in_sample_mse,
        'In_Sample_Accuracy': in_sample_accuracy,
        'Out_Sample_R2': out_of_sample_r2,
        'Out_Sample_MSE': out_of_sample_mse,
        'Out_Sample_Accuracy': out_of_sample_accuracy,
        'In_Sample_Predictions': in_sample_predictions,
        'Out_Sample_Predictions': out_of_sample_predictions,
        'Shap_Results': shap_results,  # All SHAP details as a nested dictionary
        'shap_values': shap_results['shap_values'] if shap_results else None,
        'mean_shap': shap_results['mean_shap'] if shap_results else None,
        'importance_df': importance_df,
    
        'out_sample_correlations' :shap_out_sample_correlations if shap_results else None,
        'Best_Params': best_params if model_type in ['nn_1_layer', 'nn_2_layer'] else None}


    
def calculate_z_score_with_factor(data, window_range):
    result = pd.DataFrame(index=data.index)
    
    for window in window_range:
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
    
        # Calculate Z-score
        z_score = (data - rolling_mean) / rolling_std
        result[f"Z_Score_Window_{window}"] = z_score
    
        
        factor = z_score.copy()
        factor[z_score < 2] = 1  # Set factor to 1 if Z-score < 2
        factor[z_score >= 2] = 1 + (z_score[z_score >= 2] - 2)  # Add the excess over 2 std deviations
        result[f"Factor_Window_{window}"] = factor
    
        return result


def meta_model_mass_study(
    asset,
    training_data_size,
    out_of_sample_size,
    start_date,
    end_date,
    data,
    lookback_windows,
    forecast_window,
    gnn_features=False,
    dynamic_vol_update='NONE',
    model_type='linear',
    remove_corr=True, 
    autoencoders='true',
    selection='pca'):
    # Ensure data is within the specified date range
    data = data.loc[start_date:end_date]
    print("..gen vol features")

    data[f'{asset}_percent_returns'] = data[f'{asset}_adj_price'].pct_change() * 100  
    
    
    # Generate Volatility-Based Features
    vol_features = generate_vol_features(data, asset, lookback_windows)

    # Dynamically adjust lookback and forecast windows if required - need to update the vol impact and trigger
    vol_scale =  calculate_z_score_with_factor(data, window_range=1000)
    # need to update above below is which factor or all to update form vol period - but reduce vol to smaller size to capture more recent market moves over 2 stdv
    if dynamic_vol_update == 'all':
        vol_scale =  calculate_z_score_with_factor(data, window_range=1000)
        adjusted_lookback_windows = [int(window / vol_scale) for window in lookback_windows]
        forecast_window = int(forecast_window / vol_scale)
        training_data_size = int(training_data_size / vol_scale)
    elif dynamic_vol_update == 'TW':
             
        vol_scale =  calculate_z_score_with_factor(data, window_range=1000)
        training_data_size = int(training_data_size / vol_scale)
    elif dynamic_vol_update == 'FEW':
        vol_scale =  calculate_z_score_with_factor(data, window_range=1000)     
        adjusted_lookback_windows = [int(window / vol_scale) for window in lookback_windows]
    elif dynamic_vol_update == 'FW':
        vol_scale =  calculate_z_score_with_factor(data, window_range=1000)      
        forecast_window = int(forecast_window / vol_scale)
        adjusted_lookback_windows = lookback_windows            
    elif dynamic_vol_update == 'NONE':
        adjusted_lookback_windows = lookback_windows

    # Generate Features
    if gnn_features:
        print("..gen reversion features with gnn")
        reversion_features = generate_revserion_features(data, asset, lookback_windows, gnn_features=gnn_features)
    else:
        print("..gen reversion features")
        reversion_features = generate_revserion_features(data, asset, lookback_windows)

    print("..gen trend features")
    trend_features = generate_trend_features(data, asset, adjusted_lookback_windows)
    print("..gen noise features")
    noise_features = generate_noise_reduction_features(data, asset, adjusted_lookback_windows,autoencoders_=autoencoders)
    print("..gen target")
    target = (data[f"{asset}_adj_price"].shift(-forecast_window) - data[f"{asset}_adj_price"]) / data[f"{asset}_adj_price"]

    # Drop rows with NaN values across features and target
    combined_df = pd.concat([trend_features, reversion_features, noise_features, vol_features, target], axis=1).dropna()

    # Ensure start_date and end_date are in the index
    if start_date not in combined_df.index:
        print(f"Start date {start_date} is not in the index. Using the nearest available date: {combined_df.index.min()}")
        start_date = combined_df.index[combined_df.index.get_indexer([start_date], method="nearest")[0]]

    if end_date not in combined_df.index:
        print(f"End date {end_date} is not in the index. Using the nearest available date: {combined_df.index.max()}")
        end_date = combined_df.index[combined_df.index.get_indexer([end_date], method="nearest")[0]]

    # Convert start_date and end_date to integer indices
    start_date_idx = combined_df.index.get_loc(start_date)
    end_date_idx = combined_df.index.get_loc(end_date)

    # Ensure start_date_idx and end_date_idx are integers
    if isinstance(start_date_idx, slice):
        start_date_idx = start_date_idx.start
    if isinstance(end_date_idx, slice):
        end_date_idx = end_date_idx.stop - 1

    results = {}
    results_df = pd.DataFrame()
    
    # Replace `inf` and `-inf` with `NaN` - check whats causing 
    trend_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Forward fill the `NaN` values
    trend_features.ffill(inplace=True)
    
    

    
    # Rolling window 
    count = 0
    while start_date_idx + training_data_size + out_of_sample_size <= end_date_idx:
        count += 1
        print(f"..data roll {count}")
        print(start_date_idx)
        
        # Define training and testing ranges
        training_period = pd.DateOffset(months=training_data_size)
        testing_period = pd.DateOffset(months=out_of_sample_size)
    
        # Calculate start and end dates for train and test
        train_start_date = combined_df.index[start_date_idx]
        train_end_date = train_start_date + training_period - pd.Timedelta(minutes=1)
        test_start_date = train_end_date + pd.Timedelta(minutes=1)
        test_end_date = test_start_date + testing_period - pd.Timedelta(minutes=1)
    
        # Adjust test_start_date and test_end_date if exceeding the data range
        if test_start_date > combined_df.index[-1]:
            print(f"No valid test data available starting from {test_start_date}. Stopping rolls.")
            break
    
        if test_end_date > combined_df.index[-1]:
            print(f"Adjusting test_end_date to fit within available data: {combined_df.index[-1]}")
            test_end_date = combined_df.index[-1]
    
        # Select train and test data
        train_data = combined_df.loc[train_start_date:train_end_date]
        test_data = combined_df.loc[test_start_date:test_end_date]
    
        # Check if test_data is valid; if not, stop the loop
        if test_data.empty:
            print(f"No valid test data for the range {test_start_date} to {test_end_date}. Stopping rolls.")
            break

        
        print(f"Training Data: {train_start_date} to {train_end_date}")
        print(f"Testing Data: {test_start_date} to {test_end_date}")

        # Align features to the rolling window
        trend_features_train_data = trend_features.reindex(train_data.index)
        trend_features_out_sample_data = trend_features.reindex(test_data.index)
        reversion_features_train_data = reversion_features.reindex(train_data.index)
        reversion_features_out_sample_data = reversion_features.reindex(test_data.index)
        noise_features_train_data = noise_features.reindex(train_data.index)
        noise_features_out_sample_data = noise_features.reindex(test_data.index)
        vol_features_train_data = vol_features.reindex(train_data.index)
        vol_features_out_sample_data = vol_features.reindex(test_data.index)
        target_train_data = target.reindex(train_data.index)
        target_out_sample_data = target.reindex(test_data.index)
        

        num_inf_values = np.isinf(trend_features_train_data).sum().sum()
        print(f"Number of infinity values in trend_features_train_data: {num_inf_values}")
        num_inf_values = np.isinf(trend_features_out_sample_data).sum().sum()
        print(f"Number of infinity values in ttrend_features_out_sample_data: {num_inf_values}")    
        trend_features_train_data[np.isinf(trend_features_train_data)] = 0
        trend_features_out_sample_data[np.isinf(trend_features_out_sample_data)] = 0
            
        
        print("..feature selection")
        if selection == "pca":
            selected_trend_features , trend_features_out_sample_data = apply_pca(trend_features_train_data, trend_features_out_sample_data)
            selected_noise_features , noise_features_out_sample_data = apply_pca(noise_features_train_data, noise_features_out_sample_data)
            selected_vol_features , vol_features_out_sample_data = apply_pca(vol_features_train_data, vol_features_out_sample_data)
            selected_reversion_features , reversion_features_out_sample_data = apply_pca(reversion_features_train_data, reversion_features_out_sample_data)

        else:
            selected_trend_features, trend_top_features = select_features(trend_features_train_data, target_train_data, selection=selection)
            selected_reversion_features, reversion_top_features = select_features(reversion_features_train_data, target_train_data, selection=selection)
            selected_noise_features, noise_top_features = select_features(noise_features_train_data, target_train_data, selection=selection)
            selected_vol_features, vol_top_features = select_features(vol_features_train_data, target_train_data, selection=selection)

            trend_features_out_sample_data = trend_features_out_sample_data[trend_top_features]
            reversion_features_out_sample_data = reversion_features_out_sample_data[reversion_top_features]
            noise_features_out_sample_data = noise_features_out_sample_data[noise_top_features]
            vol_features_out_sample_data = vol_features_out_sample_data[vol_top_features]

        
        # Fit the models
        trend_results = fit_model(
            train_features=selected_trend_features,
            train_target=target_train_data,
            test_features=trend_features_out_sample_data,
            test_target=target_out_sample_data,
            model_type=model_type
        )
        

        
        # Create a DataFrame for trend results and concatenate
        trend_results_df = pd.DataFrame([{
            'Asset': asset,
            'Lookback_Window': lookback_windows,
            'Forecast_Window': forecast_window,
            'Training_Size': training_data_size,
            'Out_Sample_Size': out_of_sample_size,
            'Dynamic_Vol_Update': dynamic_vol_update,
            'Feature_Selection': selection,
            'Model_Type': model_type,
            'features': 'trend',
            'roll': count,
            'Shap_Results': trend_results['Shap_Results'],
            'Shap_values': trend_results['shap_values'],
            
            'mean_shap': trend_results['mean_shap'],
            'out_sample_correlation': trend_results['out_sample_correlations'], 
            'importance_df': trend_results['importance_df'],
            
            'In_Sample_R2': trend_results['In_Sample_R2'],
            'In_Sample_MSE': trend_results['In_Sample_MSE'],
            'In_Sample_Accuracy': trend_results['In_Sample_Accuracy'],
            'Out_Sample_R2': trend_results['Out_Sample_R2'],
            'Out_Sample_MSE': trend_results['Out_Sample_MSE'],
            'Out_Sample_Accuracy': trend_results['Out_Sample_Accuracy']
        }])
        results_df = pd.concat([results_df, trend_results_df], ignore_index=True)
        
        reversion_results = fit_model(
            train_features=selected_reversion_features,
            train_target=target_train_data,
            test_features=reversion_features_out_sample_data,
            test_target=target_out_sample_data,
            model_type=model_type
        )
        
        # Create a DataFrame for reversion results and concatenate
        reversion_results_df = pd.DataFrame([{
            'Asset': asset,
            'Lookback_Window': lookback_windows,
            'Forecast_Window': forecast_window,
            'Training_Size': training_data_size,
            'Out_Sample_Size': out_of_sample_size,
            'Dynamic_Vol_Update': dynamic_vol_update,
            'Feature_Selection': selection,
            'Model_Type': model_type,
            'features': 'reversion',
            'roll': count,
            'Shap_Results': reversion_results['Shap_Results'],
            'Shap_values': reversion_results['shap_values'],
            
            'mean_shap': reversion_results['mean_shap'],
            'out_sample_correlation': reversion_results['out_sample_correlations'],   
            'importance_df': reversion_results['importance_df'],
            
            'In_Sample_R2': reversion_results['In_Sample_R2'],
            'In_Sample_MSE': reversion_results['In_Sample_MSE'],
            'In_Sample_Accuracy': reversion_results['In_Sample_Accuracy'],
            'Out_Sample_R2': reversion_results['Out_Sample_R2'],
            'Out_Sample_MSE': reversion_results['Out_Sample_MSE'],
            'Out_Sample_Accuracy': reversion_results['Out_Sample_Accuracy']
        }])
        results_df = pd.concat([results_df, reversion_results_df], ignore_index=True)
        
        noise_results = fit_model(
            train_features=selected_noise_features,
            train_target=target_train_data,
            test_features=noise_features_out_sample_data,
            test_target=target_out_sample_data,
            model_type=model_type
        )
        
        # Create a DataFrame for noise results and concatenate
        noise_results_df = pd.DataFrame([{
            'Asset': asset,
            'Lookback_Window': lookback_windows,
            'Forecast_Window': forecast_window,
            'Training_Size': training_data_size,
            'Out_Sample_Size': out_of_sample_size,
            'Dynamic_Vol_Update': dynamic_vol_update,
            'Feature_Selection': selection,
            'Model_Type': model_type,
            'features': 'noise',
            'roll': count,
            'Shap_Results': noise_results['Shap_Results'],
            'Shap_values': noise_results['shap_values'],
            
            'mean_shap': noise_results['mean_shap'],
            'out_sample_correlation': noise_results['out_sample_correlations'],  
            'importance_df': noise_results['importance_df'],
            
            
            'In_Sample_R2': noise_results['In_Sample_R2'],
            'In_Sample_MSE': noise_results['In_Sample_MSE'],
            'In_Sample_Accuracy': noise_results['In_Sample_Accuracy'],
            'Out_Sample_R2': noise_results['Out_Sample_R2'],
            'Out_Sample_MSE': noise_results['Out_Sample_MSE'],
            'Out_Sample_Accuracy': noise_results['Out_Sample_Accuracy']
        }])
        results_df = pd.concat([results_df, noise_results_df], ignore_index=True)
        
        vol_results = fit_model(
            train_features=selected_vol_features,
            train_target=target_train_data,
            test_features=vol_features_out_sample_data,
            test_target=target_out_sample_data,
            model_type=model_type
        )
        
        # Create a DataFrame for vol results and concatenate
        vol_results_df = pd.DataFrame([{
            'Asset': asset,
            'Lookback_Window': lookback_windows,
            'Forecast_Window': forecast_window,
            'Training_Size': training_data_size,
            'Out_Sample_Size': out_of_sample_size,
            'Dynamic_Vol_Update': dynamic_vol_update,
            'Feature_Selection': selection,
            'Model_Type': model_type,
            'features': 'vol',
            'roll': count,
            'Shap_Results': vol_results['Shap_Results'],
            'Shap_values': vol_results['shap_values'],
            
            'mean_shap': vol_results['mean_shap'],
            'out_sample_correlation': vol_results['out_sample_correlations'],   
            'importance_df': vol_results['importance_df'],
            
            
            'In_Sample_R2': vol_results['In_Sample_R2'],
            'In_Sample_MSE': vol_results['In_Sample_MSE'],
            'In_Sample_Accuracy': vol_results['In_Sample_Accuracy'],
            'Out_Sample_R2': vol_results['Out_Sample_R2'],
            'Out_Sample_MSE': vol_results['Out_Sample_MSE'],
            'Out_Sample_Accuracy': vol_results['Out_Sample_Accuracy']
        }])
        results_df = pd.concat([results_df, vol_results_df], ignore_index=True)
        
        # Combine Predictions for Meta-Model
        meta_model_features_train = pd.concat([
            pd.DataFrame(trend_results['In_Sample_Predictions'], index=trend_features_train_data.index, columns=['Trend_Predictions']),
            pd.DataFrame(reversion_results['In_Sample_Predictions'], index=reversion_features_train_data.index, columns=['Reversion_Predictions']),
            pd.DataFrame(noise_results['In_Sample_Predictions'], index=noise_features_train_data.index, columns=['Noise_Predictions']),
            pd.DataFrame(vol_results['In_Sample_Predictions'], index=vol_features_train_data.index, columns=['Vol_Predictions'])
        ], axis=1)
        
        meta_model_features_out_sample = pd.concat([
            pd.DataFrame(trend_results['Out_Sample_Predictions'], index=trend_features_out_sample_data.index, columns=['Trend_Predictions']),
            pd.DataFrame(reversion_results['Out_Sample_Predictions'], index=reversion_features_out_sample_data.index, columns=['Reversion_Predictions']),
            pd.DataFrame(noise_results['Out_Sample_Predictions'], index=noise_features_out_sample_data.index, columns=['Noise_Predictions']),
            pd.DataFrame(vol_results['Out_Sample_Predictions'], index=vol_features_out_sample_data.index, columns=['Vol_Predictions'])
        ], axis=1)
        
        # Fit Meta-Model
        meta_results = fit_model(
            train_features=meta_model_features_train,
            train_target=target_train_data,
            test_features=meta_model_features_out_sample,
            test_target=target_out_sample_data,
            model_type=model_type
        )
        
        # Create a DataFrame for meta-model results and concatenate
        meta_results_df = pd.DataFrame([{
            'Asset': asset,
            'Lookback_Window': lookback_windows,
            'Forecast_Window': forecast_window,
            'Training_Size': training_data_size,
            'Out_Sample_Size': out_of_sample_size,
            'Dynamic_Vol_Update': dynamic_vol_update,
            'Feature_Selection': selection,
            'Model_Type': model_type,
            'features': 'meta',
            'roll': count,
            'Shap_Results': meta_results['Shap_Results'],
            'Shap_values': meta_results['shap_values'],
            
            'mean_shap': meta_results['mean_shap'],
            'out_sample_correlation': meta_results['out_sample_correlations'],      
            'importance_df': meta_results['importance_df'],
            
            'In_Sample_R2': meta_results['In_Sample_R2'],
            'In_Sample_MSE': meta_results['In_Sample_MSE'],
            'In_Sample_Accuracy': meta_results['In_Sample_Accuracy'],
            'Out_Sample_R2': meta_results['Out_Sample_R2'],
            'Out_Sample_MSE': meta_results['Out_Sample_MSE'],
            'Out_Sample_Accuracy': meta_results['Out_Sample_Accuracy']
        }])
        results_df = pd.concat([results_df, meta_results_df], ignore_index=True)
        
        # Save Results
        results[(asset, forecast_window)] = {
            'Meta_Model_Results': meta_results,
            'Meta_Model_Out_Sample_Predictions': meta_results['Out_Sample_Predictions'],
            'Out_Sample_Target': target_out_sample_data
        }
        
        


        start_date_idx = combined_df.index[start_date_idx] + pd.DateOffset(months=out_of_sample_size)
        start_date_idx  = combined_df.index[combined_df.index.get_indexer([start_date_idx ], method="nearest")[0]]
        start_date_idx = combined_df.index.get_loc(start_date_idx)
        print(start_date_idx)

    return results, results_df, trend_features, trend_features_train_data, selected_trend_features , trend_features_out_sample_data, meta_model_features_train






# 1 layer network - stability checks acorss neurons, ensemble, seed, epochs, - heat map - ideally some break down on weights and bring back information onwhat feature causes the output the most
# reduce overfitting
# next steps - Self-Attention Mechanism: Incorporate transformer-based attention to weigh historical data points dynamically, focusing on those most relevant to current trends.
# also a additional language model to potentially decipher trained on market data
#meta model to select on above
# rolling standard deviation track on what it predicts - classification
# re-classify and group data - labelling correctly
# range forecasting window and maybe dynamically select on largest diff on stdv roll - if above a certain window choose the given forecast time frame with said selected mean reversal/trend etc




