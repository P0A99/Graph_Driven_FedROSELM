import numpy as np
import time
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import r2_score
import random
import pandas as pd
import os

def geometric_median(X, eps=1e-6):
    tic =time.time()
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        a = W * X[nonzeros]
        T = np.sum(a, 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            toc = time.time()
            timee = toc-tic
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            toc = time.time()
            timee = toc-tic
            return y1

        y = y1


def chunk_generator(X, y, n_hidden, chunk_size_0=1000, chunk_size=1, mode='basic', optional_next_dataset=None,Max=400,Min=50,mixed_difficult_percent=None):
    if mixed_difficult_percent == None:
        mixed_difficult_percent = 0.5
    N = len(y)
    if mode == 'basic':
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        return batches_x, batches_y
    elif mode == 'csfc':
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size_0, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size_0, N, chunk_size)]
        batches_x[0] = X[:chunk_size_0]
        batches_y[0] = y[:chunk_size_0]
        return batches_x, batches_y
    elif mode == 'co':
        batches_x = [X[:chunk_size_0]]
        batches_y = [y[:chunk_size_0]]
        for i in np.arange(0,chunk_size_0,chunk_size):
            batches_x.append(X[i:i+chunk_size])
            batches_y.append(y[i:i+chunk_size])
            return batches_x, batches_y
    elif mode == 'TOT':
        batches_x = [X[0:chunk_size]]
        batches_y = [y[0:chunk_size]]
        return batches_x, batches_y
    elif mode =='basic_scheduler':
        difficulty_measure = ((denormalize(y,Max,Min) < 70) | (denormalize(y,Max,Min) > 180)).astype(int)
        X_difficult = X[difficulty_measure[:,0] == 1]
        y_difficult = y[difficulty_measure == 1].reshape((X_difficult.shape[0],1))
        X_easy = X[difficulty_measure[:,0] != 1]
        y_easy = y[difficulty_measure != 1].reshape((X_easy.shape[0],1))
        N_difficult = len(y_difficult)
        N_easy = len(y_easy)
        '''print('il numero di campioni difficili è: '+str(N_difficult))
        print('il numero di campioni facili è: '+str(N_easy))'''
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_x_difficult = [X_difficult[:n_hidden]] + [X_difficult[i:i+chunk_size] for i in np.arange(chunk_size, N_difficult, chunk_size)]
        batches_y_difficult = [y_difficult[:n_hidden]] + [y_difficult[i:i+chunk_size] for i in np.arange(chunk_size, N_difficult, chunk_size)]
        batches_x_easy = [X_easy[:n_hidden]] + [X_easy[i:i+chunk_size] for i in np.arange(chunk_size, N_easy, chunk_size)]
        batches_y_easy = [y_easy[:n_hidden]] + [y_easy[i:i+chunk_size] for i in np.arange(chunk_size, N_easy, chunk_size)]
        return batches_x, batches_y, batches_x_difficult, batches_y_difficult, batches_x_easy, batches_y_easy
    elif mode == 'percent_scheduler':
        difficulty_measure = ((denormalize(y,Max,Min) < 70) | (denormalize(y,Max,Min) > 180)).astype(int)
        X_difficult = X[difficulty_measure[:,0] == 1]
        y_difficult = y[difficulty_measure == 1].reshape((X_difficult.shape[0],1))
        X_easy = X[difficulty_measure[:,0] != 1]
        y_easy = y[difficulty_measure != 1].reshape((X_easy.shape[0],1))
        '''print('il numero di campioni difficili è: '+str(N_difficult))
        print('il numero di campioni facili è: '+str(N_easy))'''
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        X_mixed = mix_lists(X_difficult,X_easy,mixed_difficult_percent)
        y_mixed = mix_lists(y_difficult,y_easy,mixed_difficult_percent)
        N_mixed = len(y_mixed)
        batches_x_mixed = [X_mixed[:n_hidden]] + [X_mixed[i:i+chunk_size] for i in np.arange(chunk_size, N_mixed, chunk_size)]
        batches_y_mixed = [y_mixed[:n_hidden]] + [y_mixed[i:i+chunk_size] for i in np.arange(chunk_size, N_mixed, chunk_size)]
        return batches_x, batches_y, batches_x_mixed, batches_y_mixed
    elif mode == 'incremental_diff':
        difficulty_measure = ((denormalize(y,Max,Min) < 70) | (denormalize(y,Max,Min) > 180)).astype(int)
        X_difficult = X[difficulty_measure[:,0] == 1]
        y_difficult = y[difficulty_measure == 1].reshape((X_difficult.shape[0],1))
        X_easy = X[difficulty_measure[:,0] != 1]
        y_easy = y[difficulty_measure != 1].reshape((X_easy.shape[0],1))
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        N_tot = ((len(y_easy) + len(y_difficult))//2)//chunk_size
        local_percent = 0
        batches_x_mixed = []
        batches_y_mixed = []
        X_semplici = X_easy.copy()
        y_semplici = y_easy.copy()
        X_difficili = X_difficult.copy()
        y_difficili = y_difficult.copy()
        for i in range(N_tot):
            num_difficili = (i * chunk_size) // N_tot
            num_difficili = min(num_difficili, chunk_size // 2)
            num_semplici = chunk_size - num_difficili
            batch_X_semplici = X_semplici[:num_semplici]
            batch_y_semplici = y_semplici[:num_semplici]
            batch_X_difficili = X_difficili[:num_difficili]
            batch_y_difficili = y_difficili[:num_difficili]
            X_semplici = X_semplici[num_semplici:]
            y_semplici = y_semplici[num_semplici:]
            X_difficili = X_difficili[num_difficili:]
            y_difficili = y_difficili[num_difficili:]
            batch_X_corrente = np.concatenate((batch_X_semplici, batch_X_difficili), axis=0)
            batch_y_corrente = np.concatenate((batch_y_semplici, batch_y_difficili), axis=0)
            np.random.shuffle(batch_X_corrente)
            np.random.shuffle(batch_y_corrente)
            batches_x_mixed.append(batch_X_corrente)
            batches_y_mixed.append(batch_y_corrente)
        return batches_x, batches_y, batches_x_mixed, batches_y_mixed
    elif mode == 'triple_regressor':   
        ipo_measure = denormalize(y,Max,Min) < 100
        iper_measure = denormalize(y,Max,Min) > 200
        X_ipo = X[ipo_measure[:,0]]
        y_ipo = y[ipo_measure].reshape((X_ipo.shape[0],1))
        X_iper = X[iper_measure[:,0]]
        y_iper = y[iper_measure].reshape((X_iper.shape[0],1))
        N_ipo = len(y_ipo)
        N_iper = len(y_iper)
        batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(chunk_size, N, chunk_size)]
        batches_x_ipo = [X_ipo[:n_hidden]] + [X_ipo[i:i+chunk_size] for i in np.arange(chunk_size, N_ipo, chunk_size)]
        batches_y_ipo = [y_ipo[:n_hidden]] + [y_ipo[i:i+chunk_size] for i in np.arange(chunk_size, N_ipo, chunk_size)]
        batches_x_iper = [X_iper[:n_hidden]] + [X_iper[i:i+chunk_size] for i in np.arange(chunk_size, N_iper, chunk_size)]
        batches_y_iper = [y_iper[:n_hidden]] + [y_iper[i:i+chunk_size] for i in np.arange(chunk_size, N_iper, chunk_size)]
        return batches_x, batches_y, batches_x_ipo, batches_y_ipo, batches_x_iper, batches_y_iper


def fit_sequential(model, X, y, n_hidden, chunk_size_0=1, chunk_size=1):
    """Fit 'model' with data 'X' and 'y', sequentially with mini-batches of
    'chunk_size' (starting with a batch of 'n_hidden' size)"""
    # Sequential learning
    N = len(y)
    # The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
    batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
    batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
    

    for b_x, b_y in zip(batches_x, batches_y):
            model.fit(b_x, b_y)

    return model

def RMSE(output, target):
    metric = 0
    for i in range(output.shape[0]):
        metric += (output[i]-target[i])**2
    metric = np.sqrt(metric/output.shape[0])
    return metric

def MAE(output, target):
    metric = 0
    for i in range(output.shape[0]):
        metric += abs(output[i]-target[i])
    metric = metric/output.shape[0]
    return metric

def Rsquared(output, target):
    metric = r2_score(target,output)
    return metric

def aggregator(clients, mode='mean', scores=None, iteration=None, hidden_clients=None, Similarity=None, Cluster=None, Betas_in=None, index=None):
    sum_w = 0
    sum_scores = 0
    beta_glob = 0
    if hidden_clients == None:
        if mode == 'mean':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]
            beta_glob = beta_glob/len(clients)
            return beta_glob
            
        elif mode == 'len_weighted':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]*clients[idx][8]
                sum_w += clients[idx][8]
            beta_glob = beta_glob/sum_w
            return beta_glob
            
        elif mode == 'score_weighted':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]*(1/scores[idx][iteration])
                sum_scores += 1/scores[idx][iteration]
            beta_glob = beta_glob/sum_scores
            return beta_glob
            
        elif mode == 'GM':
            beta_dim = clients[0][9].shape[0]
            num_clients = len(clients)
            betas_matrix = np.zeros((num_clients,beta_dim))
            for idx, client in enumerate(clients):
                beta_T = np.transpose(client[9])
                betas_matrix[idx] = beta_T
            beta_glob = np.transpose(geometric_median(betas_matrix)).reshape((beta_dim,1))
            return beta_glob
        elif mode == 'cluster_mean':
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)

            cluster1 = np.array(df.iloc[[0,1,2,4,5,8,11], [1,2,3,5,6,9,12]].values)
            cluster2 = np.array(df.iloc[[3,6,7,9,10], [4,7,8,10,11]].values)
            sum_cluster1 = sum(cluster1)
            sum_cluster2 = sum(cluster2)
            if Cluster == 1:
                for idx, client in enumerate(clients):
                    beta_glob += sum_cluster1[idx]*clients[idx][9]
                beta_glob = beta_glob/sum(sum_cluster1)
            else:
                for idx, client in enumerate(clients):
                    beta_glob += sum_cluster2[idx]*clients[idx][9]
                beta_glob = beta_glob/sum(sum_cluster2)
            return beta_glob
        elif mode == 'similarity':
            betas = [[] for _ in range(12)]
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)
            matrix = np.array(df.iloc[:,1:].values)
            for i in range(matrix.shape[0]):
                beta_glob = 0
                for idx, client in enumerate(clients):
                    beta_glob += matrix[i, idx]*Betas_in[idx][0]
                beta_glob = beta_glob/sum(matrix[i])
                betas[i] = beta_glob
            return betas
        elif mode == 'similarity_second':
            betas = [[] for _ in range(12)]
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)
            matrix = np.array(df.iloc[:,1:].values)

            for i in range(matrix.shape[0]):
                beta_glob = 0
                for idx, client in enumerate(clients):
                    temp = Betas_in[idx, i]
                    beta_glob += matrix[idx, i]*temp
                beta_glob = beta_glob/sum(matrix[i])
                betas[i] = beta_glob
            return betas
        else:
            Warning('Aggregation mode not recognized')
    else:
        if type(hidden_clients) == int: 
            hidden_clients = [hidden_clients]
        clients = [elemento for indice, elemento in enumerate(clients) if indice not in hidden_clients] 
        if mode == 'mean':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]
            beta_glob = beta_glob/len(clients)
            return beta_glob
            
        elif mode == 'len_weighted':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]*clients[idx][8]
                sum_w += clients[idx][8]
            beta_glob = beta_glob/sum_w
            return beta_glob
            
        elif mode == 'score_weighted':
            for idx, client in enumerate(clients):
                beta_glob += clients[idx][9]*(1/scores[idx][iteration])
                sum_scores += 1/scores[idx][iteration]
            beta_glob = beta_glob/sum_scores
            return beta_glob
            
        elif mode == 'GM':
            beta_dim = clients[0][9].shape[0]
            num_clients = len(clients)
            betas_matrix = np.zeros((num_clients,beta_dim))
            for idx, client in enumerate(clients):
                beta_T = np.transpose(client[9])
                betas_matrix[idx] = beta_T
            beta_glob = np.transpose(geometric_median(betas_matrix)).reshape((beta_dim,1))
            return beta_glob
        elif mode == 'cluster_mean':
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)

            cluster1 = np.array(df.iloc[[0,1,2,4,5,8,11], [1,2,3,5,6,9,12]].values)
            cluster2 = np.array(df.iloc[[3,6,9,10], [4,7,10,11]].values)
            sum_cluster1 = sum(cluster1)
            sum_cluster2 = sum(cluster2)
            if Cluster == 1:
                for idx, client in enumerate(clients):
                    beta_glob += sum_cluster1[idx]*clients[idx][9]
                beta_glob = beta_glob/sum(sum_cluster1)
            else:
                for idx, client in enumerate(clients):
                    beta_glob += sum_cluster2[idx]*clients[idx][9]
                beta_glob = beta_glob/sum(sum_cluster2)
            return beta_glob
        elif mode == 'similarity':
            betas = [[] for _ in range(12)]
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)
            matrix = np.array(df.iloc[:,1:].values)
            for i in range(matrix.shape[0]):
                for idx, client in enumerate(clients):
                    beta_glob += matrix[i, idx]*Betas_in[idx][0]
                beta_glob = beta_glob/sum(matrix[i])
                betas[i] = beta_glob
            return betas
        elif mode == 'similarity_second':
            betas = [[] for _ in range(12)]
            current_dir = os.path.dirname(__file__)
            excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
            df = pd.read_csv(excel_path)
            matrix = np.array(df.iloc[:,1:].values)

            for i in range(matrix.shape[0]):
                beta_glob = 0
                for idx, client in enumerate(clients):
                    beta_glob += matrix[i, idx]*Betas_in[i,idx]
                beta_glob = beta_glob/sum(matrix[i])
                betas[i] = beta_glob
            return betas
        else:
            Warning('Aggregation mode not recognized')

def denormalize(input,max,min=0,mode=None):
    if mode == None:
        mode = "max_min"
    #metric = metric*(max-min) + min
    if mode == "max_min":
        out = input*(max-min) + min
        return out

def test_metrics(model, input=None, target=None, max=None, min=None, metric_used='RMSE'):
    if metric_used == 'RMSE':
        out = denormalize(model.predict(input),max,min)
        tgt = denormalize(target,max,min)
        metric = RMSE(out,tgt)
        return metric,out,tgt
    elif metric_used == 'MAE':
        out = denormalize(model.predict(input),max,min)
        tgt = denormalize(target,max,min)
        metric = MAE(out,tgt)
        return metric,out,tgt
    elif metric_used == 'R^2':
        out = denormalize(model.predict(input),max,min)
        tgt = denormalize(target,max,min)
        metric = Rsquared(tgt,out)
        return metric,out,tgt
    elif metric_used == 'RMSE_single':
        out = denormalize(model.predict(input),max,min)
        tgt = denormalize(target,max,min)
        metric = np.abs(out-tgt)
        return metric,out,tgt
    elif metric_used == 'triple_RMSE':
        RMSE_list = []
        out = []
        tgt = []
        for i in range(input.shape[0]-1):
            cond = False
            elementi_sotto_soglia = np.mean(denormalize(input[i:i+1],max,min), axis = 1)<100
            elementi_sopra_soglia = np.mean(denormalize(input[i:i+1],max,min), axis = 1)>200
            if any(elementi_sopra_soglia):
                correct_model = model[2]
                #cond = True
                a = 'model_iper'
            elif any(elementi_sotto_soglia):
                correct_model = model[1]
                a = 'model_ipo'
            else:
                correct_model = model[0]
                a = 'model_std'
            out_temp = denormalize(correct_model.predict(input[i:i+1]),max,min)
            tgt_temp = denormalize(target[i:i+1],max,min)
            RMSE_list.append(RMSE(out_temp,tgt_temp))
            out.append(out_temp[0][0])
            tgt.append(tgt_temp[0][0])
            #print(str(tgt_temp)+ '___'+ a)
        metric = np.array(RMSE_list)
        return RMSE_list,out,tgt
    
def model_select(models,input,target,max,min):
    out = np.zeros((input.shape[0],1))
    tgt = np.zeros((input.shape[0],1))
    for i in range(input.shape[0]):
        cond = False
        elementi_sotto_soglia = np.mean(denormalize(input[i:i+1],max,min), axis = 1)<100
        elementi_sopra_soglia = np.mean(denormalize(input[i:i+1],max,min), axis = 1)>200
        if any(elementi_sopra_soglia):
            correct_model = models[2]
            cond = True
        if (any(elementi_sotto_soglia) and (cond == False)):
            correct_model = models[1]
        else:
            correct_model = models[0]
        out_temp = denormalize(correct_model.predict(input[i:i+1]),max,min)
        tgt_temp = denormalize(target[i:i+1],max,min)
        out[i] = out_temp[0,0]
        tgt[i] = tgt_temp[0,0]
    return out,tgt

def mix_lists(lista_a, lista_b, percentuale_difficili, seed=None):
    if seed is not None:
        random.seed(seed)
    percentuale_facili = 1-percentuale_difficili
    totale_elementi = min(int(round(len(lista_a)*percentuale_difficili)), int(round(len(lista_b)*percentuale_facili)))
    
    elementi_difficili = int(round(totale_elementi * percentuale_difficili))
    elementi_facili = int(round(totale_elementi * percentuale_facili))
    
    elementi_difficili = min(elementi_difficili, len(lista_a))
    elementi_facili = min(elementi_facili, len(lista_b))
    
    mixato = lista_a[:elementi_difficili] + lista_b[:elementi_facili]
    
    random.shuffle(mixato)
    
    return mixato