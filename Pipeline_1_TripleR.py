####################################
#  USE ENV Python 3.7 
####################################

import os
import numpy as np
import pandas as pd
from utils_dataset import dataset_funcs as df
from utils_dataset.dataset_access import save_data_as_npy
from utils_model.model_funcs import *
import time 
import matplotlib.pyplot as plt
from pyoselm_master.pyoselm.regul_oselm import OSELMRegressor
from Clarke_Error_Grid import clarke_error_grid
import random

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

#test = []
test1 = []
test2 = []
Dataset = np.load('glucose_level.npy')
Dataset_te = np.load('glucose_level_test.npy')
Dataset = df.Dataset_mat(Dataset)
Dataset_te = df.Dataset_mat(Dataset_te)
n_hidden = 100
chunk_size = 100
PH = 5
SW_step = 1
mode = 'basic'

online_ys = []
online_ts = []

clientsOG = []
clients_ipo = []
clients_iper = []
models = [[[],[],[]], [[],[],[]]]

plot_data = False
# Iterate over each client in the dataset
paz_i = 0
for client in range(np.load('glucose_level.npy').shape[0]):
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
    if paz_i==0 or paz_i==1 or paz_i==2 or paz_i==4 or paz_i==5 or paz_i==8 or paz_i==11 :
        test1.append([X_te, y_te])
    else:
        test2.append([X_te, y_te])
    paz_i += 1
triple_clients_data = [clientsOG, clients_ipo, clients_iper] 

# ELM Params:
activ_func = 'relu'
rs_list = [random.randint(0,200) for i in range(1)]
C = 0.01
N_possible_rounds = []
N_possible_rounds = []
for idx,client in enumerate(clientsOG):
    N_possible_rounds.append(len(client[0]))
N_rounds = np.min(np.array(N_possible_rounds))-2
agg_rule = 'GM'

# Initialize metrics and parameters
Client_performance = True
metric = []
metric_online = []
metric_round = np.zeros((len(clientsOG),1))
metric_round_online = np.zeros((len(clientsOG),1))
train_time = np.zeros((1,len(rs_list)))
beta_glob = 0
Batch_train_time = np.zeros((N_rounds,1))
clients_metrics_std = []
clients_metrics_online_std = []

#cluster vars
metric_online_cluster1 = []
clients_metrics_online_std_cluster1 = []
metric_cluster1 = []
clients_metrics_std_cluster1 = []
metric_online_cluster2 = []
clients_metrics_online_std_cluster2 = []
metric_cluster2 = []
clients_metrics_std_cluster2 = []

client_metrics = [[] for _ in range(12)]
client_metrics_cluster1  = [[] for _ in range(7)]
client_metrics_cluster2  = [[] for _ in range(5)]

#Cluster division
Cluster1 = [[clientsOG[0], clients_ipo[0], clients_iper[0]], [clientsOG[1], clients_ipo[1], clients_iper[1]], [clientsOG[2], clients_ipo[2], clients_iper[2]], [clientsOG[4], clients_ipo[4], clients_iper[4]], [clientsOG[5], clients_ipo[5], clients_iper[5]], [clientsOG[8], clients_ipo[8], clients_iper[8]], [clientsOG[11], clients_ipo[11], clients_iper[11]]]
Cluster2 = [[clientsOG[3], clients_ipo[3], clients_iper[3]], [clientsOG[6], clients_ipo[6], clients_iper[6]], [clientsOG[7], clients_ipo[7], clients_iper[7]], [clientsOG[9], clients_ipo[9], clients_iper[9]], [clientsOG[10], clients_ipo[10], clients_iper[10]]]
Clusters = [Cluster1, Cluster2]

# Train OSELMRegressor model
start = time.time()
for cluster_idx, Cluster in enumerate(Clusters):
    for rs_idx,rs in enumerate(rs_list):
        for Round in range(N_rounds):
            print('Round: ' +str(Round))
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

                # Initialize metrics and parameters
                Client_performance = True

                # Train OSELMRegressor model
                for rs_idx, rs in enumerate(rs_list):
                    start = time.time()
                    time_round_start = time.time()
                    if Round == 0:
                        # Round 0: Initialize models and metrics
                        for idx, client in enumerate(Cluster):
                            client = client[regressor_idx]
                            if not(len(client[0]) <= Round+1):
                                models[cluster_idx][regressor_idx] = OSELMRegressor(n_hidden=n_hidden,
                                                        activation_func=activ_func,
                                                        random_state=rs,
                                                        C=C)
                                models[cluster_idx][regressor_idx].fit(client[0][Round], client[1][Round])
                                beta = models[cluster_idx][regressor_idx].get_beta_params()
                                Cluster[idx][regressor_idx].append(beta)

                        Cluster_temp = []
                        for i in range(len(Cluster)):
                            Cluster_temp.append(Cluster[i][regressor_idx])    
                        beta_glob = aggregator(clients=Cluster_temp, mode=agg_rule, scores=None, iteration=Round)

                    else:
                        for idx, client in enumerate(Cluster):
                            client = client[regressor_idx]
                            if not(len(client[0]) <= Round+1):
                            # Round > 0: Update models and metricsfor idx, client in enumerate(Cluster):
                                models[cluster_idx][regressor_idx] = OSELMRegressor(n_hidden=n_hidden,
                                                        activation_func=activ_func,
                                                        random_state=rs,
                                                        C=C,
                                                        beta=beta_glob)
                                models[cluster_idx][regressor_idx].fit(client[0][Round], client[1][Round])
                                beta = models[cluster_idx][regressor_idx].get_beta_params()
                                Cluster[idx][regressor_idx][9] = beta
                        Cluster_temp = []
                        for i in range(len(Cluster)):
                            Cluster_temp.append(Cluster[i][regressor_idx])  
                        beta_glob = aggregator(clients=Cluster_temp, mode=agg_rule, scores=None, iteration=Round)  
                
                models[cluster_idx][regressor_idx] = OSELMRegressor(n_hidden=n_hidden, activation_func=activ_func, random_state=rs, C=C, beta=beta_glob)
            time_round_stop = time.time()
            Batch_train_time[Round] = time_round_stop - time_round_start
            stop = time.time()
            complexity_folder = os.path.dirname(os.path.realpath(__file__))
            # Crea il percorso completo del file in cui salvare il modello
            model_path = os.path.join(complexity_folder, 'OS-ELM.joblib')

            if Cluster == Cluster1:
                metric_round_cluster = []
                # Testing on the entire test-set
                for idx,client in enumerate(Cluster):
                    client = client[0]
                    metric_client,y,t = test_metrics(models[cluster_idx],input=test1[idx][0],target=test1[idx][0],max=max_tr,min=min_tr,metric_used='triple_RMSE')
                    metric_round_cluster.append(np.mean(metric_client))
                    client_metrics_cluster1[idx].append(np.mean(metric_client))
                metric_cluster1.append(sum(metric_round_cluster)/len(metric_round_cluster))
                clients_metrics_std_cluster1.append(np.std(metric_round_cluster))
            else:
                metric_round_cluster = []
                # Testing on the entire test-set
                for idx,client in enumerate(Cluster):
                    client = client[0]
                    metric_client,y,t = test_metrics(models[cluster_idx],input=test2[idx][0],target=test2[idx][0],max=max_tr,min=min_tr,metric_used='triple_RMSE')
                    metric_round_cluster.append(np.mean(metric_client))
                    client_metrics_cluster2[idx].append(np.mean(metric_client))
                metric_cluster2.append(sum(metric_round_cluster)/len(metric_round_cluster))
                clients_metrics_std_cluster2.append(np.std(metric_round_cluster))
########################      
# METRICS SAVES  
# ######################

df_cluster1 = pd.DataFrame(client_metrics_cluster1)
df_cluster1.to_excel('Cluster1.xlsx', index=False)

df_cluster1 = pd.DataFrame(clients_metrics_std_cluster1)
df_cluster1.to_excel('Cluster1_std.xlsx', index=False)

df_cluster2 = pd.DataFrame(client_metrics_cluster2)
df_cluster2.to_excel('Cluster2.xlsx', index=False)

df_cluster2 = pd.DataFrame(clients_metrics_std_cluster2)
df_cluster2.to_excel('Cluster2_std.xlsx', index=False)

results_folder = os.path.join(current_folder,'Risultati')
results_folder = os.path.join(results_folder,'Framework 1 v2')

saves = False
if mode == 'basic':
    folder_name = 'Run_data'
else:
    folder_name = 'Run_data'
results_folder = os.path.join(results_folder,folder_name)
if saves:
    if not os.path.exists(results_folder):
        # Crea la cartella
        os.mkdir(results_folder)
    
    np.save(os.path.join(results_folder, "tempo.npy"),train_time)   
    np.save(os.path.join(results_folder, "RMSE.npy"),metric) 
    np.save(os.path.join(results_folder, "RMSE_online.npy"),metric_online)
    np.save(os.path.join(results_folder, "RMSExClientOnline.npy"),metric_round_online)
    np.save(os.path.join(results_folder, "RMSExClient.npy"),metric_round)

########################      
# PLOTS     
# ######################
metric_round = []
for idx,client in enumerate(Cluster1):
    metric_client,y,t = test_metrics(models[0],input=test1[idx][0],target=test1[idx][0],max=max_tr,min=min_tr,metric_used='triple_RMSE')
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
plt.style.use('default')
plt.figure(figsize=(6,6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts[20:-20])),error_percent[20:-20],color='r',linewidth=0.8)
#plt.boxplot(gruppi, positions=[round(pos) for pos in positions], widths=0.7)
plt.xlabel('BGL Refernce Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.title('Pipeline 1, Ohio Dataset')
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
plt.title('Pipeline 1, Ohio Dataset')
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

intervallo_idx = (np.array(sorted_ts)>90) & (np.array(sorted_ts)<250)
intervallo = np.mean(error_percent[intervallo_idx])
print("L'errore percentuale medio commesso tra 90 e 250 mg/dL è: " + str(intervallo))

plt.style.use('default')
plt, zone = clarke_error_grid(online_ts, online_ys, '')
plt.title('Pipeline 1, Ohio Dataset')
plt.show()

zone_percent = []
for zona in zone:
    zona = (zona*100)/sum(zone)
    zone_percent.append(zona)
df = pd.DataFrame(zone_percent)
df.to_excel('CEG_prcent_cluster1.xlsx', index=False)

online_ts = []
online_ys = []
metric_round = []
for idx,client in enumerate(Cluster2):
    metric_client,y,t = test_metrics(models[1],input=test2[idx][0],target=test2[idx][0],max=max_tr,min=min_tr,metric_used='triple_RMSE')
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
plt.style.use('default')
plt.figure(figsize=(6,6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts[20:-20])),error_percent[20:-20],color='r',linewidth=0.8)
plt.xlabel('BGL Refernce Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.title('Pipeline 1, Ohio Dataset')
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
plt.title('Pipeline 1, Ohio Dataset')
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

intervallo_idx = (np.array(sorted_ts)>90) & (np.array(sorted_ts)<250)
intervallo = np.mean(error_percent[intervallo_idx])
print("L'errore percentuale medio commesso tra 90 e 250 mg/dL è: " + str(intervallo))

plt.style.use('default')
# PLOT CLARKE ERROR GRID
plt, zone = clarke_error_grid(online_ts, online_ys, '')
plt.title('Pipeline 1, Ohio Dataset')
plt.show()

zone_percent = []
for zona in zone:
    zona = (zona*100)/sum(zone)
    zone_percent.append(zona)
df = pd.DataFrame(zone_percent)
df.to_excel('CEG_prcent_cluster2.xlsx', index=False)

plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.plot(metric_cluster1, color='red', label='test on test set', linewidth=0.6)
plt.fill_between(range(N_rounds), np.array(metric_cluster1) - np.array(clients_metrics_std_cluster1), np.array(metric_cluster1) + np.array(clients_metrics_std_cluster1), color='red', alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.title('Pipeline 1, Ohio Dataset')
plt.grid(True)
plt.legend()
plt.show()

plt.style.use('default')
plt.figure(figsize=(6, 6))
plt.plot(metric_cluster2, color='red', label='test on test set', linewidth=0.6)
plt.fill_between(range(N_rounds), np.array(metric_cluster2) - np.array(clients_metrics_std_cluster2), np.array(metric_cluster2) + np.array(clients_metrics_std_cluster2), color='red', alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.title('Pipeline 1, Ohio Dataset')
plt.grid(True)
plt.legend()
plt.show()
