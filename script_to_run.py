#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:15:19 2025

@author: jasmincoughtrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:02:07 2025

@author: jasmincoughtrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:51:59 2025

@author: jasmincoughtrey
"""


#from submodel_functions import (generate_minute_gnn_features, apply_submodels, GCN)
#from gen_fake_data import generate_fake_data
from gen_gnn_data import GCN, generate_minute_gnn_features
from meta_model import meta_model_mass_study
#from analyse_results import analyse_results, compare_sharpe_significance
# from backtest import backtest_multiple_portfolios_with_vol_adjustment_and_costs
import pandas as pd
from data_analysis import initial_analysis
from joblib import dump, load
# # load correct data

#data = generate_fake_data()
data = pd.read_parquet("merged_assets.parquet")
# generate gnn data initially then save and download via pickle vs running each time




# mass study
# parameters to run
#window ranges intraday to weekly
look_back_window_mass_study = [
    [5, 15, 60, 240, 480, 480*5],
    [5, 30, 60, 480, 480*2, 480*10],
    [10, 20, 120, 360, 720, 1440]]
# forecast_windows = range(5,800,15)
#look_back_window_mass_study = [[10,100,300,4200]]
forecast_windows = range(60,601,60)
start_date = "2022-01-01"
end_date = "2023-12-31"
autoencoder = True
feature_selection = {'pca','tree','f_test'}
model_types = {'linear' ,'lasso','tree','boosted_tree', 'nn_1_layer','nn_2_layer'}
#feature_selection = {'tree'}
#model_types = {'linear'}
training_data_size = 6
out_of_sample_size = 3
aggregated_results_df = pd.DataFrame()
use_gnn = False # or add gnn feature
if use_gnn:
    assets = [f"asset{i}" for i in range(1, 11)]  # 10 assets
    gnn_model = GCN(input_dim=1, hidden_dim=10, output_dim=4)
    # Generate rolling GNN features (60-minute rolling window)
    gnn_features = generate_minute_gnn_features(data, assets, gnn_model, window_size_minutes=600)
    initial_analysis(data, gnn_results=gnn_features, rare_event_threshold=3)

# can set to all - 'FEW' = features, 'FW' - forecast window - 'TW' = training window
# at some point look to add in MES as a feedback loop too

autoencoders_input ='true'
dynamic_vol_update='NONE'

assets = ["ES", "HG", "MES", "MNQ", "ZF", "ZN", "ZT", "GC"]

assets = ["ES"]
feature_selection = {'pca'}
model_types = {'linear'}
forecast_windows = range(60,61,60)
look_back_window_mass_study = [[5]]

#stats = initial_analysis(data, gnn_results=None, rare_event_threshold=3)
print('..starting loop')
all_results = {}
for forecast_window in forecast_windows:
    for i, selection in enumerate(feature_selection):
        for model_type in model_types:
            for asset in assets: 
                for lookback_windows in look_back_window_mass_study:
                    print(f"asset{asset}")
                   # Apply submodels for the current lookback window set
                    results, results_df,target_x, meta_model_prediction_x  = meta_model_mass_study(
                        asset,
                        training_data_size,
                        out_of_sample_size,
                        start_date,
                        end_date,
                        data,
                        lookback_windows,
                        forecast_window,
                        gnn_features=gnn_features if use_gnn else False,  # Pass gnn_features explicitly
                        dynamic_vol_update='NONE',  
                        model_type=model_type,       
                        remove_corr=True,   
                        autoencoders= autoencoders_input,
                        selection=selection)
                    

                  #   Store results
                    all_results[f"Lookback_Set_{i + 1}"] = {'asset': asset,'Lookback_Windows': lookback_windows,'model_type':model_type,'feature_selection': selection,'forecast_window ':forecast_window, 'Results': results, 'tagret' : target_x, 'predictions' : meta_model_prediction_x}
                
                    aggregated_results_df = pd.concat([aggregated_results_df, results_df], ignore_index=True)


#aggregated_results_df.to_pickle("aggregated_results_df.pkl")
#all_results.to_pickle("all_results.pkl")
# # start date minus decided training date window potentially
start_date = "2024-01-01"
end_date = "2024-12-31"
# # analyse best resudump(all_results, "all_results.pkl")lts for parameters and stability and use on complete out of sample               
#best_params, aggregated_importance = analyse_results(assets,df_loaded,all_results)
# final_results = {}
# for asset in assets:
#      final_results[asset] = meta_model_mass_study(asset=asset,training_data_size=training_data_size,out_of_sample_size=out_of_sample_size,start_date=start_date,end_date=end_date ,data=data,lookback_windows=best_params[asset]['Lookback_Set'],forecast_window=[best_params[asset]['Forecast_Window']],dynamic_vol_update=False, model_type=best_params[asset]['Model_Type'],selection=best_params[asset]['Feature_Selection'])

# compare statistical significance of confidence interval of sharpe 
# comparison = compare_sharpe_significance(results, final_results, confidence=0.95)

# run backtest
# strategies, sharpe_ratios = backtest_multiple_portfolios_with_vol_adjustment_and_costs(final_results, data, assets, target_vol=0.01, vol_window=252)
