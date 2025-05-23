####################################
#  USE ENV Python 3.7 
####################################

import os
import sys
import joblib
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from utils_dataset import dataset_funcs as df
from utils_dataset.dataset_access import save_data_as_npy
from utils_model.model_funcs import *
import time 
import matplotlib.pyplot as plt
from pyoselm_master.pyoselm.regul_oselm import OSELMRegressor
from Clarke_Error_Grid import clarke_error_grid
import random
import math
from sklearn.metrics import matthews_corrcoef

plt.rcParams.update({'font.size': 14})
plt.style.use('seaborn-darkgrid')

#GET PAZs PATH
def load_datasets():
    """
    Loads the datasets from the 'ohio_datasets' folder.

    Returns:
        paz_list_train (list): List of file paths for training datasets.
        paz_list_test (list): List of file paths for testing datasets.
    """
    current_folder = os.path.dirname(__file__)
    datasets_folder = os.path.join(current_folder, 'ohio_datasets')
    datasets_train_folder = os.path.join(datasets_folder, 'Train')
    datasets_test_folder = os.path.join(datasets_folder, 'Test')
    paz_list_train = []
    paz_list_test = []
    for paz_name in os.listdir(datasets_train_folder):
        path = os.path.join(datasets_train_folder, paz_name)
        if os.path.isfile(path):
            paz_list_train.append(path)
    for paz_name in os.listdir(datasets_test_folder):
        path = os.path.join(datasets_test_folder, paz_name)
        if os.path.isfile(path):
            paz_list_test.append(path)

    return paz_list_train, paz_list_test,current_folder


########################      
# ACCESS TO DATASET     
########################
paz_list_train, paz_list_test, current_folder = load_datasets()
save_data_as_npy(current_folder,paz_list_train,paz_list_test) 
current_folder = os.path.dirname(__file__)


#######################
# LOADING DATA     
#######################

test1 = []  # Group 1 test data
test2 = []  # Group 2 test data

# Load saved datasets (in .npy format)
Dataset = np.load('glucose_level.npy')
Dataset_te = np.load('glucose_level_test.npy')

# Convert datasets into structured objects for easier handling
Dataset = df.Dataset_mat(Dataset)
Dataset_te = df.Dataset_mat(Dataset_te)

# Configuration parameters
n_hidden = 100         # Number of hidden neurons in the model
chunk_size = 100       # Size of each training chunk
PH = 5                 # Prediction horizon
SW_step = 1            # Step size for the sliding window
mode = 'basic'         # Chunking mode

# Initialize containers
online_ys = []         # Online model predictions
online_ts = []         # Ground truth values for online predictions

clientsOG = []         # Original normalization
clients_ipo = []       # Initial point normalization
clients_iper = []      # Peak value normalization
models = [[[],[],[]], [[],[],[]]]  # Models for Cluster 1 and Cluster 2

plot_data = False
paz_i = 0  # Patient index

# Iterate through all patients and preprocess their data
for client in range(np.load('glucose_level.npy').shape[0]):

    # Extract training and testing data with associated normalization values
    X_tr, y_tr, max_it_tr, min_it_tr = Dataset.dataset_to_train(
        paz=client, PH=PH, train_seq=5, train_freq=SW_step
    )
    X_te, y_te, max_it_te, min_it_te = Dataset_te.dataset_to_train(
        paz=client, PH=PH, train_seq=5, train_freq=SW_step
    )

    # Generate chunks using basic mode
    bX_tr, by_tr = chunk_generator(X_tr, y_tr, n_hidden, chunk_size=chunk_size, mode=mode)
    bX_te, by_te = chunk_generator(X_te, y_te, n_hidden, chunk_size=chunk_size, mode=mode)

    # Generate chunks using triple regressor mode (original, IPO, IPER)
    bX_tr, by_tr, bX_tr_ipo, by_tr_ipo, bX_tr_iper, by_tr_iper = chunk_generator(
        X_tr, y_tr, n_hidden, chunk_size=chunk_size, mode='triple_regressor', Max=max_it_tr, Min=min_it_tr
    )
    bX_te, by_te, bX_te_ipo, by_te_ipo, bX_te_iper, by_te_iper = chunk_generator(
        X_te, y_te, n_hidden, chunk_size=chunk_size, mode='triple_regressor', Max=max_it_te, Min=min_it_te
    )

    # Structure client data for the three normalization modes
    client_data =      [bX_tr,      by_tr,      bX_te,      by_te,      max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]
    client_data_ipo =  [bX_tr_ipo,  by_tr_ipo,  bX_te_ipo,  by_te_ipo,  max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]
    client_data_iper = [bX_tr_iper, by_tr_iper, bX_te_iper, by_te_iper, max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]

    # Append processed data to respective lists
    clientsOG.append(client_data)
    clients_ipo.append(client_data_ipo)
    clients_iper.append(client_data_iper)

    # Divide patients into two test groups (for separate evaluation)
    if paz_i in [0, 1, 2, 4, 5, 8, 11]:
        test1.append([X_te, y_te])
    else:
        test2.append([X_te, y_te])
    
    paz_i += 1

# Combine data for all normalization strategies
triple_clients_data = [clientsOG, clients_ipo, clients_iper]
########################
# MODEL TRAIN     
# ######################

# ELM Hyperparameters
activ_func = 'relu'
rs_list = [random.randint(0, 200) for _ in range(1)]  # Random seed list
C = 0.01                                              # Regularization coefficient

# Determine number of rounds based on the shortest client dataset
N_possible_rounds = []
for idx, client in enumerate(clientsOG):
    N_possible_rounds.append(len(client[0]))  # Number of training chunks
N_rounds = np.min(np.array(N_possible_rounds)) - 2
N_rounds = 10

# Aggregation strategy for federated training
agg_rule = 'cluster_mean'

# Initialize containers for performance metrics
Client_performance = True
metric = []
metric_online = []
metric_round = np.zeros((len(clientsOG), 1))
metric_round_online = np.zeros((len(clientsOG), 1))
train_time = np.zeros((1, len(rs_list)))
Batch_train_time = np.zeros((N_rounds, 1))
beta_glob = 0

# Metrics per cluster
metric_online_cluster1 = []
clients_metrics_online_std_cluster1 = []
metric_cluster1 = []
clients_metrics_std_cluster1 = []

metric_online_cluster2 = []
clients_metrics_online_std_cluster2 = []
metric_cluster2 = []
clients_metrics_std_cluster2 = []

# Per-client performance tracking
client_metrics = [[] for _ in range(12)]
client_metrics_cluster1 = [[] for _ in range(7)]
client_metrics_cluster2 = [[] for _ in range(5)]

# Predefined cluster assignments
Cluster1 = [
    [clientsOG[0], clients_ipo[0], clients_iper[0]],
    [clientsOG[1], clients_ipo[1], clients_iper[1]],
    [clientsOG[2], clients_ipo[2], clients_iper[2]],
    [clientsOG[4], clients_ipo[4], clients_iper[4]],
    [clientsOG[5], clients_ipo[5], clients_iper[5]],
    [clientsOG[8], clients_ipo[8], clients_iper[8]],
    [clientsOG[11], clients_ipo[11], clients_iper[11]]
]
Cluster2 = [
    [clientsOG[3], clients_ipo[3], clients_iper[3]],
    [clientsOG[6], clients_ipo[6], clients_iper[6]],
    [clientsOG[7], clients_ipo[7], clients_iper[7]],
    [clientsOG[9], clients_ipo[9], clients_iper[9]],
    [clientsOG[10], clients_ipo[10], clients_iper[10]]
]
Clusters = [Cluster1, Cluster2]

# Begin training loop for each cluster
start = time.time()
for cluster_idx, Cluster in enumerate(Clusters):

    # Identify cluster ID (used for aggregation function)
    clustering = 1 if Cluster == Cluster1 else 2

    for rs_idx, rs in enumerate(rs_list):
        for Round in range(N_rounds):
            print(f'Round: {Round}')
            time_round_start = time.time()

            for regressor_idx, clients in enumerate(triple_clients_data):

                # Compute mean normalization values across clients
                max_tr = np.mean([client[4] for client in clients])
                min_tr = np.mean([client[6] for client in clients])
                max_te = np.mean([client[5] for client in clients])
                min_te = np.mean([client[7] for client in clients])

                ############## MODEL TRAINING ##############
                if Round == 0:
                    # Round 0: Initialize and train local models
                    for idx, client in enumerate(Cluster):
                        client_data = client[regressor_idx]
                        if len(client_data[0]) > Round + 1:
                            models[cluster_idx][regressor_idx] = OSELMRegressor(
                                n_hidden=n_hidden,
                                activation_func=activ_func,
                                random_state=rs,
                                C=C
                            )
                            models[cluster_idx][regressor_idx].fit(client_data[0][Round], client_data[1][Round])
                            beta = models[cluster_idx][regressor_idx].get_beta_params()
                            Cluster[idx][regressor_idx].append(beta)

                    # Aggregate beta parameters across clients
                    Cluster_temp = [Cluster[i][regressor_idx] for i in range(len(Cluster))]
                    beta_glob = aggregator(clients=Cluster_temp, mode=agg_rule, scores=None, iteration=Round, Cluster=clustering)

                else:
                    # Round > 0: Continue training with global beta initialization
                    for idx, client in enumerate(Cluster):
                        client_data = client[regressor_idx]
                        if len(client_data[0]) > Round + 1:
                            models[cluster_idx][regressor_idx] = OSELMRegressor(
                                n_hidden=n_hidden,
                                activation_func=activ_func,
                                random_state=rs,
                                C=C,
                                beta=beta_glob
                            )
                            models[cluster_idx][regressor_idx].fit(client_data[0][Round], client_data[1][Round])
                            beta = models[cluster_idx][regressor_idx].get_beta_params()
                            Cluster[idx][regressor_idx][9] = beta  # Update global beta

                    Cluster_temp = [Cluster[i][regressor_idx] for i in range(len(Cluster))]
                    beta_glob = aggregator(clients=Cluster_temp, mode=agg_rule, scores=None, iteration=Round, Cluster=clustering)

                # Store final global model
                models[cluster_idx][regressor_idx] = OSELMRegressor(
                    n_hidden=n_hidden,
                    activation_func=activ_func,
                    random_state=rs,
                    C=C,
                    beta=beta_glob
                )

            # Store training duration for this round
            time_round_stop = time.time()
            Batch_train_time[Round] = time_round_stop - time_round_start

            # Save model path (if needed later)
            complexity_folder = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(complexity_folder, 'OS-ELM.joblib')

            # Model evaluation on test set
            if Cluster == Cluster1:
                metric_round_cluster = []
                for idx, client in enumerate(Cluster):
                    test_data = test1[idx]
                    metric_client, y, t = test_metrics(
                        models[cluster_idx],
                        input=test_data[0],
                        target=test_data[0],  # NOTE: may need to be test_data[1]
                        max=max_tr,
                        min=min_tr,
                        metric_used='triple_RMSE'
                    )
                    metric_round_cluster.append(np.mean(metric_client))
                    client_metrics_cluster1[idx].append(np.mean(metric_client))
                metric_cluster1.append(np.mean(metric_round_cluster))
                clients_metrics_std_cluster1.append(np.std(metric_round_cluster))

            else:
                metric_round_cluster = []
                for idx, client in enumerate(Cluster):
                    test_data = test2[idx]
                    metric_client, y, t = test_metrics(
                        models[cluster_idx],
                        input=test_data[0],
                        target=test_data[0],  # NOTE: may need to be test_data[1]
                        max=max_tr,
                        min=min_tr,
                        metric_used='triple_RMSE'
                    )
                    metric_round_cluster.append(np.mean(metric_client))
                    client_metrics_cluster2[idx].append(np.mean(metric_client))
                metric_cluster2.append(np.mean(metric_round_cluster))
                clients_metrics_std_cluster2.append(np.std(metric_round_cluster))
########################      
# METRICS SAVES  
# ######################

# Save per-client RMSE metrics and standard deviations for both clusters
df_cluster1 = pd.DataFrame(client_metrics_cluster1)
df_cluster1.to_excel('Cluster1.xlsx', index=False)

df_cluster1_std = pd.DataFrame(clients_metrics_std_cluster1)
df_cluster1_std.to_excel('Cluster1_std.xlsx', index=False)

df_cluster2 = pd.DataFrame(client_metrics_cluster2)
df_cluster2.to_excel('Cluster2.xlsx', index=False)

df_cluster2_std = pd.DataFrame(clients_metrics_std_cluster2)
df_cluster2_std.to_excel('Cluster2_std.xlsx', index=False)

# Define result folder path
results_folder = os.path.join(current_folder, 'Risultati', 'Framework 1 v2')
folder_name = 'Run_data'  # You can add logic here if you have multiple modes
results_folder = os.path.join(results_folder, folder_name)

# Save metrics to .npy only if 'saves' flag is True
saves = True
if saves:
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    np.save(os.path.join(results_folder, "tempo.npy"), train_time)   
    np.save(os.path.join(results_folder, "RMSE.npy"), metric) 
    np.save(os.path.join(results_folder, "RMSE_online.npy"), metric_online)
    np.save(os.path.join(results_folder, "RMSExClientOnline.npy"), metric_round_online)
    np.save(os.path.join(results_folder, "RMSExClient.npy"), metric_round)

###########################   
# CLUSTER 1 ANALYSIS   
###########################

# Evaluate and collect predictions for Cluster 1
metric_round = []
for idx, client in enumerate(Cluster1):
    metric_client, y, t = test_metrics(models[0], input=test1[idx][0], target=test1[idx][0],
                                       max=max_tr, min=min_tr, metric_used='triple_RMSE')
    metric_round.append(np.mean(metric_client))
    online_ys.extend(y)
    online_ts.extend(t)

# Sort predictions and compute percent error
combined = sorted(zip(online_ys, online_ts), key=lambda x: x[1])
array1_sorted, array2_sorted = zip(*combined)
error = np.abs(np.array(sorted(array1_sorted)) - np.array(sorted(array2_sorted)))
error_percent = np.zeros((error.shape[0], 1))
sorted_ts = sorted(online_ts)
for i in range(error.shape[0]):
    error_percent[i] = (error[i] / array2_sorted[i]) * 100

# Plot percent error vs reference BGL values
plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts[20:-20])), error_percent[20:-20], color='r', linewidth=0.8)
plt.xlabel('BGL Reference Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.title('Pipeline 2, Ohio Dataset')
plt.show()

# Plot average error in hypo/eu/hyper-glycemia ranges
plt.figure(figsize=(6, 6))
plt.grid(True)
ipo_mask = (np.array(sorted(online_ts[20:-20])) <= 100)
eug_mask = (np.array(sorted(online_ts[20:-20])) > 100) & (np.array(sorted(online_ts[20:-20])) <= 200)
iper_mask = (np.array(sorted(online_ts[20:-20])) > 200)
ipo_mean = np.mean(error_percent[20:-20][ipo_mask])
eug_mean = np.mean(error_percent[20:-20][eug_mask])
iper_mean = np.mean(error_percent[20:-20][iper_mask])
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.array(sorted(online_ts[20:-20])), error_percent[20:-20], color='r', linewidth=0.8)
ax.set_xlabel('BGL Reference Value [mg/dL]')
ax.set_ylabel('Prediction Error [%]')
plt.title('Pipeline 2, Ohio Dataset')
ax.axvspan(40, 100, color='#cce5ff', alpha=0.4)
ax.axvspan(100, 200, color='#d4edda', alpha=0.4)
ax.axvspan(200, 400, color='#f8d7da', alpha=0.4)
ax.hlines(ipo_mean, 40, 100, color='blue', linestyle='-', linewidth=1.5, label=f'Hypo Avg: {ipo_mean:.2f}%')
ax.hlines(eug_mean, 100, 200, color='green', linestyle='-', linewidth=1.5, label=f'Eu Avg: {eug_mean:.2f}%')
ax.hlines(iper_mean, 200, 400, color='red', linestyle='-', linewidth=1.5, label=f'Hyper Avg: {iper_mean:.2f}%')
ax.legend()
plt.show()

# Mean percent error within [90, 250] mg/dL range
range_idx = (np.array(sorted_ts) > 90) & (np.array(sorted_ts) < 250)
range_error = np.mean(error_percent[range_idx])
print(f"Mean percent error between 90 and 250 mg/dL: {range_error:.2f}%")

# Clarke Error Grid for Cluster 1
plt, zone = clarke_error_grid(online_ts, online_ys, '')
plt.title('Pipeline 2, Ohio Dataset')
plt.show()
zone_percent = [(z * 100) / sum(zone) for z in zone]
pd.DataFrame(zone_percent).to_excel('CEG_prcent_cluster1.xlsx', index=False)

###########################   
# CLUSTER 2 ANALYSIS   
###########################

# Reset variables
online_ts, online_ys = [], []
metric_round = []

# Evaluate and collect predictions for Cluster 2
for idx, client in enumerate(Cluster2):
    metric_client, y, t = test_metrics(models[1], input=test2[idx][0], target=test2[idx][0],
                                       max=max_tr, min=min_tr, metric_used='triple_RMSE')
    metric_round.append(np.mean(metric_client))
    online_ys.extend(y)
    online_ts.extend(t)

# Sort and compute error
combined = sorted(zip(online_ys, online_ts), key=lambda x: x[1])
array1_sorted, array2_sorted = zip(*combined)
error = np.abs(np.array(sorted(array1_sorted)) - np.array(sorted(array2_sorted)))
error_percent = np.zeros((error.shape[0], 1))
sorted_ts = sorted(online_ts)
for i in range(error.shape[0]):
    error_percent[i] = (error[i] / array2_sorted[i]) * 100

# Plot percent error
plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts[20:-20])), error_percent[20:-20], color='r', linewidth=0.8)
plt.xlabel('BGL Reference Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.title('Pipeline 2, Ohio Dataset')
plt.show()

# Average error for range 90–250 mg/dL
range_idx = (np.array(sorted_ts) > 90) & (np.array(sorted_ts) < 250)
range_error = np.mean(error_percent[range_idx])
print(f"Mean percent error between 90 and 250 mg/dL: {range_error:.2f}%")

# Clarke Error Grid for Cluster 2
plt, zone = clarke_error_grid(online_ts, online_ys, '')
plt.title('Pipeline 2, Ohio Dataset')
plt.show()
zone_percent = [(z * 100) / sum(zone) for z in zone]
pd.DataFrame(zone_percent).to_excel('CEG_prcent_cluster2.xlsx', index=False)

###########################   
# PLOT GLOBAL METRICS  
###########################

# Plot RMSE over training rounds for Cluster 1
plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.plot(metric_cluster1, color='red', label='Cluster 1', linewidth=0.6)
plt.fill_between(range(N_rounds),
                 np.array(metric_cluster1) - np.array(clients_metrics_std_cluster1),
                 np.array(metric_cluster1) + np.array(clients_metrics_std_cluster1),
                 color='red', alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.title('Pipeline 2 - RMSE over Rounds (Cluster 1)')
plt.grid(True)
plt.legend()
plt.show()

# Plot RMSE over training rounds for Cluster 2
plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.plot(metric_cluster2, color='red', label='Cluster 2', linewidth=0.6)
plt.fill_between(range(N_rounds),
                 np.array(metric_cluster2) - np.array(clients_metrics_std_cluster2),
                 np.array(metric_cluster2) + np.array(clients_metrics_std_cluster2),
                 color='red', alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.title('Pipeline 2 - RMSE over Rounds (Cluster 2)')
plt.grid(True)
plt.legend()
plt.show()
