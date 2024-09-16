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
    # Get the current folder path
    current_folder = os.path.dirname(__file__)

    # Join the current folder path with 'ohio_datasets' folder
    datasets_folder = os.path.join(current_folder, 'ohio_datasets')

    # Join the datasets folder path with 'Train' and 'Test' folders
    datasets_train_folder = os.path.join(datasets_folder, 'Train')
    datasets_test_folder = os.path.join(datasets_folder, 'Test')

    paz_list_train = []
    paz_list_test = []

    # Iterate over the files in the 'Train' folder and append the file paths to paz_list_train
    for paz_name in os.listdir(datasets_train_folder):
        path = os.path.join(datasets_train_folder, paz_name)
        if os.path.isfile(path):
            paz_list_train.append(path)

    # Iterate over the files in the 'Test' folder and append the file paths to paz_list_test
    for paz_name in os.listdir(datasets_test_folder):
        path = os.path.join(datasets_test_folder, paz_name)
        if os.path.isfile(path):
            paz_list_test.append(path)

    return paz_list_train, paz_list_test,current_folder
        
#print(paz_list) #paz_list è una lista dei path di tutti i pazienti


########################      ACCESS TO DATASET     #################
paz_list_train, paz_list_test, current_folder = load_datasets()
save_data_as_npy(current_folder,paz_list_train,paz_list_test) 
current_folder = os.path.dirname(__file__)
#####################################################################

########################      LOADING DATA     ######################



# Define an empty list to store client data
clients = [[] for _ in range(12)]

# Define an empty list to store test data
test = []

# Load the dataset from 'glucose_level.npy' file
Dataset = np.load('glucose_level.npy')

# Load the test dataset from 'glucose_level_test.npy' file
Dataset_te = np.load('glucose_level_test.npy')

# Convert the dataset into a DataFrame using 'Dataset_mat' function
Dataset = df.Dataset_mat(Dataset)

# Convert the test dataset into a DataFrame using 'Dataset_mat' function
Dataset_te = df.Dataset_mat(Dataset_te)

online_ys = []
online_ts = []

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

online_ys = []
online_ts = []

clientsOG = []
clients_ipo = []
clients_iper = []

TEMP_BETAS = np.zeros((12,12), dtype=object)

plot_data = False
# Iterate over each client in the dataset
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
    test.append([X_te, y_te])
triple_clients_data = [clientsOG, clients_ipo, clients_iper] 

##########################################################################

########################      MODEL TRAIN     ############################
# ELM Params:
activ_func = 'relu'
rs_list = [random.randint(0,200) for i in range(1)]
C = 0.01
N_possible_rounds = []
for idx,client in enumerate(clientsOG):
    N_possible_rounds.append(len(client[0]))
N_rounds = np.min(np.array(N_possible_rounds))-2
agg_rule = 'similarity'

# Initialize metrics and parameters
Client_performance = True
metric = []
metric_online = []
metric_round = np.zeros((len(clients),1))
metric_round_online = np.zeros((len(clients),1))
performance_list = []
metric_single_online = np.zeros((100,1))
train_time = np.zeros((1,len(rs_list)))
beta_glob = 0
beta_list = []
Batch_train_time = np.zeros((N_rounds,1))
y_single =[]
clients_metrics_std = []
clients_metrics_online_std = [[] for _ in range(12)]

max_tr = 0
min_tr = 0
for client in clientsOG:
    max_tr += client[4]
    min_tr += client[6]
max_tr = max_tr / len(clientsOG)
min_tr = min_tr / len(clientsOG)

# Calculate the average maximum and minimum values for test data
max_te = 0
min_te = 0
for client in clientsOG:
    max_te += client[5]
    min_te += client[7]
max_te = max_te / len(clientsOG)
min_te = min_te / len(clientsOG)
BETAS = np.zeros((len(clientsOG), len(clientsOG), len(triple_clients_data)), dtype=object)
models = np.zeros(( len(clientsOG), len(triple_clients_data)), dtype=object)
# Train OSELMRegressor model
for rs_idx,rs in enumerate(rs_list):
    start = time.time()
    client_metrics = [[] for _ in range(12)]
    client_metrics_online = [[] for _ in range(12)]
    for Round in range(N_rounds):
        print('Round: ' +str(Round))
        for regressor_idx,clients in enumerate(triple_clients_data):
            if Round == 0:
                # Round 0: Initialize models and metrics
                for idx, client in enumerate(clients):
                    model = OSELMRegressor(n_hidden=n_hidden,
                                            activation_func=activ_func,
                                            random_state=rs,
                                            C=C)
                    model.fit(client[0][Round], client[1][Round])
                    beta = model.get_beta_params()
                    #clients[idx].append(beta)
                    TEMP_BETAS[idx,0] = beta
                #np.save('BETAS_preagg',BETAS)
                beta_glob = aggregator(clients=clients, mode='similarity', scores=client_metrics, iteration=Round, Betas_in=TEMP_BETAS)
                for idx in range(len(clients)):
                    for j in range(len(clients)):
                        BETAS[idx,j,regressor_idx] = beta_glob[j]
                #np.save('BETAS_postagg',BETAS)

            else:
                # Round > 0: Update models and metrics
                for idx, client in enumerate(clients):
                    if not(len(client[0]) <= Round+1):
                        for j in range(len(clients)):
                            model = OSELMRegressor(n_hidden=n_hidden,
                                                    activation_func=activ_func,
                                                    random_state=rs,
                                                    C=C,
                                                    beta=BETAS[idx, j, regressor_idx])
                            model.fit(client[0][Round], client[1][Round])
                            beta = model.get_beta_params()
                            TEMP_BETAS[idx,j] = beta
                if not(len(client[0]) <= Round+1):
                    beta_glob = aggregator(clients=clients, mode='similarity_second', scores=client_metrics, iteration=Round, Betas_in=TEMP_BETAS)
                    for idx,client in enumerate(clients):    
                        for j in range(len(clients)):
                            BETAS[idx, j, regressor_idx] = beta_glob[j]
        metric_round = []
        metric_round_online = []


        for idx,client in enumerate(clientsOG):
            for regressor_idx in range(3):
                models[idx, regressor_idx] = OSELMRegressor(n_hidden=n_hidden, activation_func=activ_func, random_state=rs, C=C, beta=BETAS[idx,idx,regressor_idx])
        # Testing on the entire test-set
        for idx, client in enumerate(clientsOG):
            metric_client,y,t = test_metrics(models[idx],input=test[idx][0],target=test[idx][1],max=max_te,min=min_te,metric_used='triple_RMSE')
            metric_round.append(np.mean(metric_client))
            client_metrics[idx].append(metric_client[0])
        metric.append(sum(metric_round)/len(metric_round))
        clients_metrics_std.append(np.std(metric_round))




#####################################################################

########################      METRICS     ###########################

    '''train_time[0][rs_idx] = stop-start'''
    '''print('Il tempo di addestramento è pari a: ' + str(round(train_time[0][rs_idx], 5)) + ' secondi')
    print("RMSE raggiunto con " + str(N_rounds) + " Rounds, testando sull'intero test-set  è pari a " +str(np.min(metric[:,rs_idx])))
    print("RMSE raggiunto con " + str(N_rounds) + " Rounds, testando online è pari a " +str(np.min(metric_online[:,rs_idx])))
    #print(f'Il modello addestrato pesa in RAM: {Model_Space_Complexity} bytes')'''
################################################################

########################      SAVES     ########################
for idx,client in enumerate(clients):
    df_tutti = pd.DataFrame(client_metrics[idx])
    df_tutti.to_excel('Test_usando_il_modello_'+str(idx)+'.xlsx', index=False)


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

################################################################

########################      PLOTS     ########################

for idx,client in enumerate(clientsOG):
    for regressor_idx in range(3):
        models[idx,regressor_idx] = OSELMRegressor(n_hidden=n_hidden, activation_func=activ_func, random_state=rs, C=C, beta=BETAS[idx, idx, regressor_idx])
for idx,client in enumerate(clientsOG):
    metric_client,y,t = test_metrics(models[idx],input=test[idx][0],target=test[idx][1],max=max_tr,min=min_tr,metric_used='triple_RMSE')
    metric_round.append(np.mean(metric_client))
    for n,el in enumerate(y):
        online_ys.append(el)
    for m,elem in enumerate(t):
        online_ts.append(elem)

combined = list(zip(online_ys, online_ts))

# Step 2: Ordina le coppie in base agli elementi del primo array
combined_sorted = sorted(combined, key=lambda x: x[1])

# Step 3: Dividi i risultati ordinati in due array
array1_sorted, array2_sorted = zip(*combined_sorted)

error = abs(np.array(sorted(array1_sorted)) - np.array(sorted(array2_sorted)))
error_percent = np.zeros((error.shape[0],1))
sorted_ts = sorted(online_ys)
for i in range(error.shape[0]):
    error_percent[i] = (error[i]/array2_sorted[i])*100
'''plt.figure(figsize=(6,6))
plt.plot(np.array(sorted(online_ts)),error,color='r',linewidth=0.4)
plt.xlabel('BGL Value [mg/dL]')
plt.ylabel('Prediction Error [mg/dL]')
plt.show()'''

plt.figure(figsize=(6,6))
plt.grid(True)
plt.plot(np.array(sorted(online_ts))[20:-40],error_percent[20:-40],color='r',linewidth=0.8)
plt.xlabel('BGL Refernce Value [mg/dL]')
plt.ylabel('Prediction Error [%]')
plt.show()

'''
difficulty_M_hard = []
for idx in range(len(y_single)):
    if y_single[idx]>200 or y_single[idx]<70:
        difficulty_M_hard.append(1)
    else:
        difficulty_M_hard.append(0)
plt.figure(figsize=(6,6))
plt.plot(performance_list,color='b')
#plt.plot(y_single,color='r')
plt.show()

plt.figure(figsize=(6,6))
plt.plot(difficulty_M_hard,color='b')
#plt.plot(y_single,color='r')
plt.show()


# Set the location of each subplot
loc = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2],[3,0],[3,1],[3,2]]
# PLOT CONFRONTO TRA TARGET E PRED DI OGNI CLIENT
# Create a figure with subplots
fig, axes = plt.subplots(4, 3, figsize=(10, 6))
for idx, client in enumerate(clients):
    # Prediction for plot
    out = denormalize(model.predict(test[idx][0]),client[5],client[7])
    target = denormalize(test[idx][1],client[5],client[7])
    samples = np.arange(0, int(np.array(out[int(np.round(0.3*out.shape[0])):int(np.round(0.7*out.shape[0]))]).shape[0])) * 5
    row = loc[idx][0]
    col = loc[idx][1]
    ax = axes[row, col]
    # Plot the prediction and target
    ax.plot(samples, out[int(np.round(0.3*out.shape[0])):int(np.round(0.7*out.shape[0]))], color='red', label='pred', linewidth=0.4)
    ax.plot(samples, target[int(np.round(0.3*out.shape[0])):int(np.round(0.7*out.shape[0]))], color='blue', label='target', linewidth=0.4)
    ax.set_ylabel('RMSE [mg/dL]')
    ax.set_xlabel('minutes')
plt.legend()
plt.tight_layout()
plt.title('Comparison Real BGL (target) vs Predicted BGL (Pred)')
if saves:
    plt.savefig(os.path.join(results_folder, 'Target_vs_Pred'))
plt.show()'''

#PLOT SINGOLO TRACCIATO BGL
paz = 6
bgl = denormalize(test[paz][1],clients[paz][5],clients[paz][7])
samples = np.arange(0, 100)
plt.figure(figsize=(20,6))
plt.grid(True)
plt.plot(samples,bgl[100:200], color='red', linewidth=3)
plt.show()

'''
fig, axes = plt.subplots(4, 3, figsize=(12, 8))
for idx, client in enumerate(clients):
    row = loc[idx][0]
    col = loc[idx][1]
    ax = axes[row, col]
    ax.bar(['A','B','C','D','E'], zone[idx], color='green', edgecolor='black')
    ax.set_xlabel('Zone')
    ax.set_ylabel('%')
    ax.set_title('Soggetto '+str(idx+1), fontsize = 8)
if saves:
    plt.savefig(os.path.join(results_folder, 'Error_Grid_Hist'))
plt.title('CEG')
plt.show()'''

# PLOT RMSE CON TEST ONLINE E SU TUTTO IL TEST-SET
plt.figure(figsize=(6, 6))
plt.plot(metric[3:], color='red', label='test on test set', linewidth=0.6)
plt.fill_between(range(N_rounds)[3:], np.array(metric[3:]) - np.array(clients_metrics_std[3:]), np.array(metric[3:]) + np.array(clients_metrics_std[3:]), color='red', alpha=0.2)
'''plt.plot(metric_online[idx], color='blue', label='online test', linewidth=0.6)
plt.fill_between(range(N_rounds), np.array(metric_online[idx]) - np.array(clients_metrics_online_std[idx]), np.array(metric_online[idx]) + np.array(clients_metrics_online_std[idx]), color='blue', alpha=0.2)'''
plt.xlabel('Rounds')
plt.ylabel('RMSE [mg/dL]')
plt.legend()
plt.show()

'''#PLOT DEI TEMPI DI ADDESTRAMENTO PER OGNI BATCH PER TUTTI I CLIENTS   
plt.figure(figsize=(6,6))
plt.plot(Batch_train_time,color='red', label='single batch',linewidth=0.8)
plt.plot(np.mean(Batch_train_time)*np.ones((Batch_train_time.shape[0],1)),color='blue', label='average',linewidth=0.8)
plt.xlabel('Rounds')
plt.ylabel('time (s)')
plt.title('Training time of all consecutive clients for single batch')
plt.legend()
if saves:
    plt.savefig(os.path.join(results_folder, 'Train Time'))
plt.show()'''

'''plt.style.use('default')
# PLOT CLARKE ERROR GRID
zone = np.zeros((len(clients),5))
for idx, client in enumerate(clients):
    out = model.predict(test[idx][0])
    out = denormalize(out,client[5],client[7])
    target = denormalize(test[idx][1],client[5],client[7])
    # Generate Clarke Error Grid and save zone percentages
    plt, zone[idx] = clarke_error_grid(online_ts, online_ys, '')
    zone[idx] = [(elemento / sum(zone[idx])) * 100 for elemento in zone[idx]]
    if saves:
        np.savetxt(os.path.join(results_folder, f"zone_client_{idx}.txt"), zone)
plt.title('CEG')
plt.show()'''
