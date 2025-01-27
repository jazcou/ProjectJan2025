#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:52:09 2025

@author: jasmincoughtrey
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os



def save_shap_dependence_plot(shap_values, features, feature_name, output_dir="shap_plots"):
    
    # generate and save a SHAP dependence plot for a given feature.
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    plt.figure()
    shap.dependence_plot(feature_name, shap_values, features)
    plt.title(f"SHAP Dependence Plot for {feature_name}")
    plt.savefig(f"{output_dir}/{feature_name}_dependence_plot.png")
    plt.close()



def calculate_sharpe_confidence_interval(returns, confidence=0.95):

    mean_return = np.mean(returns)
    std_error = np.std(returns) / np.sqrt(len(returns))  
    sharpe_ratio = mean_return / np.std(returns)       
    z = norm.ppf((1 + confidence) / 2)                 
    ci_lower = sharpe_ratio - z * std_error
    ci_upper = sharpe_ratio + z * std_error
    return sharpe_ratio, ci_lower, ci_upper


def analyse_results(all_results):

    # Step 1: Extract feature importance, SHAP data, and performance metrics
    all_importance_dfs = []
    all_shap_dfs = []
    performance_data = []

    for key, result in all_results.items():
        asset = result['asset']
        model_type = result['model_type']
        feature_selection = result['feature_selection']
        lookback_windows = result['Lookback_Windows']

        # Extract results from submodels
        for (asset_key, forecast_window), res in result['Results'].items():
            meta_results = res['Meta_Model_Results']
            
            # Extract feature importance
            importance_df = meta_results.get('importance_df', None)
            if importance_df is not None:
                importance_df['Asset'] = asset
                importance_df['Model_Type'] = model_type
                importance_df['Feature_Selection'] = feature_selection
                importance_df['Lookback_Windows'] = str(lookback_windows)
                importance_df['Forecast_Window'] = forecast_window
                all_importance_dfs.append(importance_df)
            
            # Extract SHAP values
            shap_results = meta_results.get('Shap_Results', None)
            if shap_results:
                shap_values = shap_results['shap_values']
                mean_shap = shap_results['mean_shap']


                shap_df = pd.DataFrame({
                    'Feature': shap_values.feature_names,
                    'Mean_SHAP': mean_shap,
                    'Asset': asset,
                    'Model_Type': model_type,
                    'Feature_Selection': feature_selection,
                    'Lookback_Windows': str(lookback_windows),
                    'Forecast_Window': forecast_window
                })
                all_shap_dfs.append(shap_df)

            # Extract performance metrics
            out_sample_r2 = meta_results.get('Out_Sample_R2', None)
            out_sample_sharpe = meta_results.get('Out_Sample_Sharpe', None)
            if out_sample_r2 is not None and out_sample_sharpe is not None:
                performance_data.append({
                    'Asset': asset,
                    'Model_Type': model_type,
                    'Feature_Selection': feature_selection,
                    'Lookback_Windows': str(lookback_windows),
                    'Forecast_Window': forecast_window,
                    'Out_Sample_R2': out_sample_r2,
                    'Out_Sample_Sharpe': out_sample_sharpe
                })

    # Combine all feature importance from reverse engineering and SHAP results
    all_importance_dfs = pd.concat(all_importance_dfs, ignore_index=True) if all_importance_dfs else None
    all_shap_dfs = pd.concat(all_shap_dfs, ignore_index=True) if all_shap_dfs else None
    performance_df = pd.DataFrame(performance_data)

    # Step 2: Determine best parameters based on Sharpe ratio
    best_params = {}
    for asset in performance_df['Asset'].unique():
        asset_df = performance_df[performance_df['Asset'] == asset]
        best_row = asset_df.loc[asset_df['Out_Sample_Sharpe'].idxmax()]  
        best_params[asset] = {
            'Model_Type': best_row['Model_Type'],
            'Feature_Selection': best_row['Feature_Selection'],
            'Lookback_Windows': eval(best_row['Lookback_Windows']),
            'Forecast_Window': best_row['Forecast_Window']
        }

    # Step 3: Aggregate feature importance globally
    if all_importance_dfs is not None:
        aggregated_importance = all_importance_dfs.groupby(['Feature', 'Feature_Set', 'Asset']).agg({
            'Importance': 'mean'  # Aggregate by mean importance
        }).reset_index()
    else:
        aggregated_importance = None

    # Step 4: Visualise feature importance (heatmaps and bar plots)
    if aggregated_importance is not None:
        feature_sets = aggregated_importance['Feature_Set'].unique()
        for feature_set in feature_sets:
            subset = aggregated_importance[aggregated_importance['Feature_Set'] == feature_set]
            pivot_table = subset.pivot(index='Feature', columns='Asset', values='Importance').fillna(0)

            # Heatmap for feature importance across assets
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Importance'}
            )
            plt.title(f"Feature Importance Heatmap for {feature_set.capitalize()} Features Across Assets")
            plt.xlabel("Assets")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.savefig(f"heatmap_{feature_set}_features_across_assets.png")
            plt.show()

            # Aggregated bar plot for feature set
            grouped_importance = subset.groupby('Feature').agg({'Importance': 'mean'}).reset_index()
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=grouped_importance,
                x='Importance',
                y='Feature',
                color='skyblue',
                order=grouped_importance.sort_values(by='Importance', ascending=False)['Feature']
            )
            plt.xlabel('Mean Importance')
            plt.ylabel('Feature')
            plt.title(f"Aggregated Feature Importance for {feature_set.capitalize()} Features")
            plt.tight_layout()
            plt.savefig(f"aggregated_importance_{feature_set}.png")
            plt.show()

    # Step 5: Visualise SHAP results
    if all_shap_dfs is not None:
        aggregated_shap = all_shap_dfs.groupby('Feature').agg({
            'Mean_SHAP': 'mean',
            'Mean_Interaction': 'mean'
        }).reset_index()

        # Global SHAP feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=aggregated_shap,
            x='Mean_SHAP',
            y='Feature',
            color='skyblue',
            order=aggregated_shap.sort_values(by='Mean_SHAP', ascending=False)['Feature']
        )
        plt.xlabel('Mean SHAP Value')
        plt.ylabel('Feature')
        plt.title('Global SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig("global_shap_feature_importance.png")
        plt.show()

        # Global SHAP interaction importance
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=aggregated_shap,
            x='Mean_Interaction',
            y='Feature',
            color='skyblue',
            order=aggregated_shap.sort_values(by='Mean_Interaction', ascending=False)['Feature']
        )
        plt.xlabel('Mean Interaction Value')
        plt.ylabel('Feature')
        plt.title('Global SHAP Interaction Importance')
        plt.tight_layout()
        plt.savefig("global_shap_interaction_importance.png")
        plt.show()

    # Step 6: Save performance and importance data to Excel
    performance_df.to_excel("performance_summary.xlsx", index=False)
    if all_importance_dfs is not None:
        all_importance_dfs.to_excel("importance_summary.xlsx", index=False)
    if all_shap_dfs is not None:
        all_shap_dfs.to_excel("shap_summary.xlsx", index=False)

    return best_params, aggregated_importance




def compare_sharpe_significance(results, final_results,  confidence=0.95):

    def calculate_sharpe_and_ci(returns):

        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        sharpe_ratio = mean_return / std_dev
        std_error = std_dev / np.sqrt(len(returns))
        z = norm.ppf((1 + confidence) / 2)
        ci_lower = sharpe_ratio - z * std_error
        ci_upper = sharpe_ratio + z * std_error
        return sharpe_ratio, ci_lower, ci_upper


    comparison_results = []

    # Loop through each asset
    for asset in results.keys():
        # Extract initial period returns
        initial_returns = results[asset]["Out_Sample_Returns"]  # Replace with correct key
        initial_sharpe, initial_ci_lower, initial_ci_upper = calculate_sharpe_and_ci(initial_returns)

        # Extract final period returns
        final_returns = final_results[asset]["Out_Sample_Returns"]  # Replace with correct key
        final_sharpe, final_ci_lower, final_ci_upper = calculate_sharpe_and_ci(final_returns)

        # add to results
        comparison_results.append({
            "Asset": asset,
            "Initial_Sharpe": initial_sharpe,
            "Initial_CI_Lower": initial_ci_lower,
            "Initial_CI_Upper": initial_ci_upper,
            "Final_Sharpe": final_sharpe,
            "Final_CI_Lower": final_ci_lower,
            "Final_CI_Upper": final_ci_upper,
            "Significant_Improvement": final_sharpe > initial_ci_upper  # Check significance
        })


    comparison_df = pd.DataFrame(comparison_results)


    print("Sharpe Ratio Comparison Between Periods:")
    print(comparison_df)


    plt.figure(figsize=(12, 6))
    for _, row in comparison_df.iterrows():
        plt.errorbar(
            [row["Asset"], row["Asset"]],
            [row["Initial_Sharpe"], row["Final_Sharpe"]],
            yerr=[[row["Initial_Sharpe"] - row["Initial_CI_Lower"], row["Final_Sharpe"] - row["Final_CI_Lower"]],
                  [row["Initial_CI_Upper"] - row["Initial_Sharpe"], row["Final_CI_Upper"] - row["Final_Sharpe"]]],
            fmt="o", capsize=5, label=f"{row['Asset']} Sharpe (Initial vs Final)"
        )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Sharpe Ratios with Confidence Intervals (Initial vs Final Year)")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Assets")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return comparison_df
