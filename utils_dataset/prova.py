import numpy as np
import math
train_seq = 5
PH = 6
train_freq = 8
index = [7,25, 96]
array = np.empty((100, 2), dtype='U10')  # 'U10' indica il tipo di dati stringa con lunghezza massima di 10 caratteri

# Riempimento della prima colonna con numeri da 0 a 99
array[:, 0] = np.arange(100).astype(str)

# Riempimento della seconda colonna con numeri da 100 a 199
array[:, 1] = np.arange(100, 200).astype(str)
values = array[:,1].astype(float)
#Features and labels inizialization
X = np.zeros((int(math.floor(array.shape[0]-train_seq-PH)/train_freq),train_seq))
y = np.zeros((int(math.floor(array.shape[0]-train_seq-PH)/train_freq),1))
if len(index) > 0:
    array_nan = values.copy()
    array_nan[index] = np.nan
    Max = np.nanmax(array_nan)
    Min = np.nanmin(array_nan)
else:
    Max = np.max(array)
    Min = np.min(array)
if len(index) > 0: #List non-empty (delay errors exists)
    i=0
    k=0
    while k <= X.shape[0]:
        for j in index:
            for z in range(train_seq+PH):
                if j == i+z:
                    i += z+1
        if (k <= X.shape[0]) and (i<=values.shape[0]-train_seq-PH):
            X[k] = values[i:i+train_seq]
            y[k] = values[i+train_seq+PH-1]
            i += train_freq
            k += 1
        else:
            i += train_freq
            k += 1
        cccc = 1
    assigned_index = (X != 0).any(axis=1)
    X_new = X[assigned_index].reshape(-1,train_seq)
    y_new = y[assigned_index].reshape(-1,1)
cc=1








        X = 1e-200*np.ones((array.shape[0]-train_seq,train_seq))
        y = 1e-200*np.ones((array.shape[0]-train_seq,1))
        data = np.zeros((array.shape[0],1))
        for it in range(len(data)):
            data[it] = float(array[it,1])
        Max = np.max(data)
        Min = np.min(data)
        
        if len(index)>0: #List non-empty (delay errors exists)
            i = 0
            k = 0
            check = 0
            while i < array.shape[0]-2*PH:
                for j in index:
                    #Here we skip the indexes that contains delay errors, X's rows must contains only samples
                    #that are time-spaced in the right way
                    for z in range(1,train_seq-1):
                        if (j == i+z):
                            i += z+1
                            check += 1
                if train_seq == 5:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
                elif train_seq == 6:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH-1)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1]), float(array[i+5,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH+1,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
                elif train_seq == 7:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH-2)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1]), float(array[i+5,1]), float(array[i+6,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH+2,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
            # We need to delete the elements that are initializated but not assigned
            # This happen when we have delay errors
            assigned_index = (X > 1e-100).any(axis=1)
            if train_seq == 5:
                X_new = X[assigned_index].reshape(-1,5)
                y_new = y[assigned_index].reshape(-1,1)
            elif train_seq == 6:
                X_new = X[assigned_index].reshape(-1,6)
                y_new = y[assigned_index].reshape(-1,1)
            elif train_seq == 7:
                X_new = X[assigned_index].reshape(-1,7)
                y_new = y[assigned_index].reshape(-1,1)
            return X_new, y_new, Max, Min
                
        else:
            i = 0
            for element in range(array.shape[0]-train_seq):
                if train_seq == 5:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
                elif train_seq == 6:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1]), float(array[i+5,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH+1,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
                elif train_seq == 7:
                    if ((k < X.shape[0]) and (i < array.shape[0]-2*PH)):
                        seq = np.array([float(array[i,1]), float(array[i+1,1]), float(array[i+2,1]), float(array[i+3,1]), float(array[i+4,1]), float(array[i+5,1]), float(array[i+6,1])])
                        seq = self.Normalize(input=seq,Max=Max,Min=Min)
                        X[k] = seq
                        target = float(array[i+2*PH+2,1])
                        target = self.Normalize(input=target,Max=Max,Min=Min)
                        y[k] = target
                        i += 1
                        k += 1
                    else:
                        i +=1
                        k += 1
            return X, y, Max, Min  