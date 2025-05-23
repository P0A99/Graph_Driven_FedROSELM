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

N_clients = 12
clientsOG = []
clients_ipo = []
clients_iper = []
models = [[],[],[]]
test = []
Dataset = np.load('glucose_level.npy')
Dataset_te = np.load('glucose_level_test.npy')
Dataset = df.Dataset_mat(Dataset)
Dataset_te = df.Dataset_mat(Dataset_te)
n_hidden = 100
chunk_size = 100
PH = 5
SW_step = 1
mode = 'basic'
activ_func = 'relu'
agg_rule = 'GM'
C = 0.01
rs_list = [random.randint(0,200) for i in range(1)]

plot_data = False

for client in range(12):
    # Get the training data for the current client
    X_tr, y_tr, max_it_tr, min_it_tr = Dataset.dataset_to_train(paz=client, PH=PH, train_seq=5, train_freq=SW_step)
    X_te, y_te, max_it_te, min_it_te = Dataset_te.dataset_to_train(paz=client,PH=PH,train_seq=5, train_freq=SW_step)
    bX_tr, by_tr = chunk_generator(X_tr, y_tr, n_hidden, chunk_size=chunk_size, mode=mode)
    bX_te, by_te = chunk_generator(X_te, y_te, n_hidden, chunk_size=chunk_size, mode=mode)
    bX_tr, by_tr, bX_tr_ipo, by_tr_ipo, bX_tr_iper, by_tr_iper = chunk_generator(X_tr, y_tr, n_hidden, chunk_size=chunk_size, mode='triple_regressor', Max=max_it_tr, Min=min_it_tr)
    bX_te, by_te, bX_te_ipo, by_te_ipo, bX_te_iper, by_te_iper = chunk_generator(X_te, y_te, n_hidden, chunk_size=chunk_size, mode='triple_regressor', Max=max_it_te, Min=min_it_te)
    client_data = [bX_tr, by_tr, bX_te, by_te, max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]
    client_data_ipo = [bX_tr_ipo, by_tr_ipo, bX_te_ipo, by_te_ipo, max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]
    client_data_iper = [bX_tr_iper, by_tr_iper, bX_te_iper, by_te_iper, max_it_tr, max_it_te, min_it_tr, min_it_te, X_tr.shape[0]]

    clientsOG.append(client_data)
    clients_ipo.append(client_data_ipo)
    clients_iper.append(client_data_iper)
    test.append([X_te, y_te])
triple_clients_data = [clientsOG, clients_ipo, clients_iper]    


#Init
N_possible_rounds = []
for idx,client in enumerate(clientsOG):
    N_possible_rounds.append(len(client[0]))
N_rounds = np.min(np.array(N_possible_rounds))-2
metric = []
metric_online = []
online_ys = []
online_ts = []
beta_glob = 0
times = np.zeros((N_rounds,2))
clients_metrics_std = []
clients_metrics_online_std = []

client_metrics = [[] for i in range(12)]

for Round in range(N_rounds):
    start1 = time.time()
    for regressor_idx,clients in enumerate(triple_clients_data):
        # Calculate the average maximum and minimum values for training data
        max_tr = 0
        min_tr = 0
        for client in clients:
            max_tr += client[4]
            min_tr += client[6]
        max_tr = max_tr / len(clients)
        min_tr = min_tr / len(clients)

        # Calculate the average maximum and minimum values for test data
        max_te = 0
        min_te = 0
        for client in clients:
            max_te += client[5]
            min_te += client[7]
        max_te = max_te / len(clients)
        min_te = min_te / len(clients)

########################
# MODEL TRAIN     
# ######################
        '''N_possible_rounds = []
        for idx,client in enumerate(clients):
            N_possible_rounds.append(len(client[0]))
        N_rounds = np.min(np.array(N_possible_rounds))-1'''

        # Initialize metrics and parameters
        Client_performance = False

        # Train OSELMRegressor model
        for rs_idx, rs in enumerate(rs_list):
            start = time.time()
            client_metrics_online = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
            time_round_start = time.time()
            if Round == 0:
                # Round 0: Initialize models and metrics
                for idx, client in enumerate(clients):
                    if not(len(client[0]) <= Round+1):
                        models[regressor_idx] = OSELMRegressor(n_hidden=n_hidden,
                                                activation_func=activ_func,
                                                random_state=rs,
                                                C=C)
                        models[regressor_idx].fit(client[0][Round], client[1][Round])
                        beta = models[regressor_idx].get_beta_params()
                        clients[idx].append(beta)
                        '''if Client_performance:
                            client_metric_online = test_metrics(models[regressor_idx],
                                                        input=client[0][Round+1],
                                                        target=client[1][Round+1],
                                                        max=client[4],
                                                        min=client[6],
                                                        metric_used='RMSE')
                            client_metrics_online[idx].append(client_metric_online)
                            client_metric = test_metrics(models[regressor_idx],
                                                        input=test[idx][0],
                                                        target=test[idx][1],
                                                        max=client[4],
                                                        min=client[6],
                                                        metric_used='RMSE')
                            client_metrics[idx].append(client_metric)'''

                beta_glob = aggregator(clients=clients, mode=agg_rule, scores=client_metrics, iteration=Round)
            else:
                # Round > 0: Update models and metrics
                for idx, client in enumerate(clients):
                    if not(len(client[0]) <= Round+1):
                        models[regressor_idx] = OSELMRegressor(n_hidden=n_hidden,
                                                activation_func=activ_func,
                                                random_state=rs,
                                                C=C,
                                                beta=beta_glob)
                        models[regressor_idx].fit(client[0][Round], client[1][Round])
                        beta = models[regressor_idx].get_beta_params()
                        clients[idx][9] = beta
                        '''if Client_performance:
                            client_metric_online = test_metrics(models[regressor_idx],
                                                        input=client[0][Round+1],
                                                        target=client[1][Round+1],
                                                        max=client[4],
                                                        min=client[6],
                                                        metric_used='RMSE')
                            client_metrics_online[idx].append(client_metric_online)
                            client_metric = test_metrics(models[regressor_idx],
                                                        input=test[idx][0],
                                                        target=test[idx][1],
                                                        max=client[4],
                                                        min=client[6],
                                                        metric_used='RMSE')
                            client_metrics[idx].append(client_metric)'''

                beta_glob = aggregator(clients=clients, mode=agg_rule, scores=client_metrics, iteration=Round)  
        
        models[regressor_idx] = OSELMRegressor(n_hidden=n_hidden, activation_func=activ_func, random_state=rs, C=C, beta=beta_glob)
        time_round_stop = time.time()
        stop = time.time()

    metric_round = []
    metric_round_online = []
    # Testing on next training chunk
    end1 = time.time()
    start2 = time.time()
    for idx,client in enumerate(clientsOG):
        metric_client_online,y_online_temp,t_online = test_metrics(models,input=client[0][Round+1],target=client[1][Round+1],max=max_tr,min=min_tr,metric_used='triple_RMSE')
        metric_round_online.append(np.mean(metric_client_online))    
    metric_online.append(sum(metric_round_online)/len(metric_round_online))
    clients_metrics_online_std.append(np.std(metric_round_online))

    # Testing on the entire test-set
    for idx,client in enumerate(clientsOG):
        metric_client,y,t = test_metrics(models,input=test[idx][0],target=test[idx][1],max=max_te,min=min_te,metric_used='triple_RMSE')
        metric_round.append(np.mean(metric_client))
        client_metrics[idx].append(np.mean(metric_client))
    metric.append(sum(metric_round)/len(metric_round))
    clients_metrics_std.append(np.std(metric_round))
    '''
    end2 = time.time()
    train_time = end1 - start1
    test_time = end2 - start2
    times[Round,0] = train_time
    times[Round,1] = test_time
    '''
    print('Round: ' +str(Round))


########################      
# METRICS SAVES  
# ######################

df = pd.DataFrame(client_metrics)
df.to_excel('Clients.xlsx', index=False)

df = pd.DataFrame(clients_metrics_std)
df.to_excel('Clients_std.xlsx', index=False)
    
for idx,client in enumerate(clientsOG):
    metric_client,y,t = test_metrics(models,input=test[idx][0],target=test[idx][1],max=max_tr,min=min_tr,metric_used='triple_RMSE')
    metric_round.append(np.mean(metric_client))
    for n,el in enumerate(y):
        online_ys.append(el)
    for m,elem in enumerate(t):
        online_ts.append(elem)
combined = list(zip(online_ys, online_ts))
combined_sorted = sorted(combined, key=lambda x: x[1])
array1_sorted, array2_sorted = zip(*combined_sorted)
error = abs(np.array(sorted(array1_sorted)) - np.array(sorted(array2_sorted)))
error_percent = np.zeros((error.shape[0],1))
sorted_ts = sorted(online_ts)
for i in range(error.shape[0]):
    error_percent[i] = (error[i]/array2_sorted[i])*100
'''plt.figure(figsize=(6,6))
plt.plot(np.array(sorted(online_ts)),error,color='r',linewidth=0.4)
plt.xlabel('BGL Value [mg/dL]')
plt.ylabel('Prediction Error [mg/dL]')
plt.show()'''


########################      
# PLOTS     
# ######################
plt.style.use('default')
plt.figure(figsize=(6,6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts[20:-20])),error_percent[20:-20],color='r',linewidth=0.8)
plt.xlabel('BGL Refernce Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.title('Pipeline 0, Ohio Dataset')
plt.show()

plt.style.use('default')
plt.figure(figsize=(6,6))
plt.grid(True)
ipo_mask = (np.array(sorted(online_ts[20:-20])) <= 100)
eug_mask = (np.array(sorted(online_ts[20:-20])) > 100) & (np.array(sorted(online_ts[20:-20])) <= 200)
iper_mask = (np.array(sorted(online_ts[20:-20])) > 200)
ipo_mean = np.mean(error_percent[20:-20][ipo_mask])
eug_mean = np.mean(error_percent[20:-20][eug_mask])
iper_mean = np.mean(error_percent[20:-20][iper_mask])
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.array(sorted(online_ts[20:-20])), error_percent[20:-20], color='r', linewidth=0.8)
ax.set_xlabel('BGL Reference Value [mg/dL]')
ax.set_ylabel('Prediction Error [%]')
plt.title('Pipeline 0, Ohio Dataset')
ax.axvspan(40, 100, color='#cce5ff', alpha=0.4 )#label='Hypoglycemia'
ax.axvspan(100, 200, color='#d4edda', alpha=0.4)#, label='Euglycemia'
ax.axvspan(200, 400, color='#f8d7da', alpha=0.4)#, label='Hyperglycemia'
#ax.axvline(x=100, color='blue', linestyle='--', label='Hypoglycemia/Euglycemia Boundary')
#ax.axvline(x=200, color='green', linestyle='--', label='Euglycemia/Hyperglycemia Boundary')
ax.hlines(ipo_mean, 40, 100, color='blue', linestyle='-', linewidth=1.5, label=f'Average Hypoglycemia Value: {ipo_mean:.2f}%')
ax.hlines(eug_mean, 100, 200, color='green', linestyle='-', linewidth=1.5, label=f'Average Euglycemia Value: {eug_mean:.2f}%')
ax.hlines(iper_mean, 200, 400, color='red', linestyle='-', linewidth=1.5, label=f'Average Hyperglycemia Value: {iper_mean:.2f}%')

ax.grid(True)
ax.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.grid(True)
plt.plot(metric, color='red', label='test on test set', linewidth=0.6)
plt.fill_between(range(N_rounds), np.array(metric) - np.array(clients_metrics_std), np.array(metric) + np.array(clients_metrics_std), color='red', alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.legend()
plt.title('Pipeline 0, Ohio Dataset')
plt.show()


plt.style.use('default')
# PLOT CLARKE ERROR GRID
plt, zone = clarke_error_grid(online_ts, online_ys, '')
plt.title('Pipeline 0, Ohio Dataset')
plt.show()

zone_percent = []
for zona in zone:
    zona = (zona*100)/sum(zone)
    zone_percent.append(zona)
df = pd.DataFrame(zone_percent)
df.to_excel('CEG_prcent.xlsx', index=False)

