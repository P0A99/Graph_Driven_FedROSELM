import numpy as np
import math
import os
import openpyxl
import pandas as pd
from datetime import datetime, timedelta

###################################        Dataset_mat CLASS        ###################################
class Dataset_mat:
    def __init__(self,array,sim=False):
        #Inizialization array: ARRAY must be the matrix of the dataset with n-row and 2-column('date','value')
        # n : number of samples
        self.array = array
        self.sim = sim
    
    def paz_ext(self,paz=0):
        #This simple function return the row of the matrix-Dataset assosiated to one patient ('paz')
        out = self.array[paz]
        return out

    def Normalize(self,input=None,Max=0,Min=0):
        if type(input) == type(None): #If nothing is passed as input, the input is the global variable 'array'
            input = self.array
        out = (input-Min)/(Max-Min)
        #out = input/Max
        return out
        
    def check_time(self,input=None,delay=5,th=1,paz=0):
        #This function check if the first column ('date') of the array correspondig to the patient ('paz') contains
        #samples that are time-spaced from each other by ('delay')('delay' is expressed in minutes)
        func_list = []
        if type(input) == type(None): #If nothing is passed as input, the input is the global variable 'array'
            input = self.array[paz]
        paz_array = input
        non_empty_elements = paz_array != ''#Checking if the array contains empty elements
        if np.any(~non_empty_elements) and self.sim == False:
            paz_new = paz_array[np.array([not(col1=='' or col2=='') for col1, col2 in paz_array])]
            '''print(f"L'array del paziente {paz} contiene elementi vuoti. "
            f"Tali elementi sono stati eliminati passando da dimensioni {paz_array.shape} "
            f"a dimensioni {paz_new.shape}.")'''
            #If the array contains empty elements, those elements are deleted and the array is reshaped
        else:
            paz_new = paz_array
            '''print(f"L'array del paziente {paz} non contiene elementi vuoti.")'''
        if self.sim == False:
            correct_delay = True    
            date_objects = [datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S') for date_str in paz_new[:,0]]
            #Checking if the variable 'paz_new' contains elements in the column ('date') of the patient array that are time-spced
            #by ('delay')
            for i in range(1, len(date_objects)):
                diff = date_objects[i] - date_objects[i - 1]
                if (diff > timedelta(minutes=delay+th)) or (diff < timedelta(minutes=delay-th)):
                    '''print(f'Errore: il campione in posizione {i} non è sfasato di {delay} minuti dal campione precedente')'''
                    correct_delay = False
                    func_list.append(i) #Here we track the indexes of the element not time-spaced by ('delay')

            if correct_delay:
                '''print('Tutti i campioni sono stati acquisiti correttamente: delay = 5 min')'''
            return paz_new, func_list
        #paz_new : the array on the patient ('paz') fixed and reshaped (it is (n,2) matrix)
        else:
            func_list = []
            return paz_new, func_list

            
    def check_value(self,input=None,paz=0):
        #This function finds missing values in the 2-th column of the paz-th row of the ARRAY
        if type(input) == type(None):
            input = self.array[paz]
        paz_array = self.check_time(input,paz=paz) #Applying check_time()
        k=0
        integrity = True
        for value in paz_array[:,1]:
            if float(value) == None:
                '''print(f"L'elemento {k} nella sequenza di valori del paziente {paz} è mancante")'''
                integrity = False
            k += 1
        if integrity:
            '''print("Tutti i campioni contengono un valore")'''
        #This function just gives a feedback and don't change the input
    
    def dataset_to_train(self,paz=0,train_seq=5,PH=5,train_freq=1):
        #This function return the paz-th row of the ARRAY transformed in the right format for the training-phase:
        # 1-th training example   [X : [1-th sample, 2-th sample, 3-th sample, 4-th sample, 5-th sample] , y : [10-th sample]
        # ...
        # ...
        # ...
        # n-th training example   [X : [n-th sample, (n+1)-th sample, (n+2)-th sample, (n+3)-th sample, (n+4)-th sample] , y : [(n+9)-th sample]
        array,index = self.check_time(paz=paz)#Applying check_time()
        if self.sim == False:
            values = array[:,1].astype(float)
        else:
            values = array.astype(float)
        #Features and labels inizialization
        X = np.zeros((int(math.floor(array.shape[0]-train_seq-PH)),train_seq))
        y = np.zeros((int(math.floor(array.shape[0]-train_seq-PH)),1))
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
                if (i <= X.shape[0]) and (i<=values.shape[0]-train_seq-PH):
                    X[k] = self.Normalize(values[i:i+train_seq],Max=Max,Min=Min)
                    y[k] = self.Normalize(values[i+train_seq+PH-1],Max=Max,Min=Min)
                    i += train_freq
                    k += 1
                else:
                    i += train_freq
                    k += 1
            assigned_index = (X != 0).any(axis=1)
            X_new = X[assigned_index].reshape(-1,train_seq)
            y_new = y[assigned_index].reshape(-1,1)
            return X_new, y_new, Max, Min
        else:
            i=0
            k=0
            while k < X.shape[0]:
                if (i <= X.shape[0]) and (i<=values.shape[0]-train_seq-PH):
                    X[k] = self.Normalize(values[i:i+train_seq],Max=Max,Min=Min)
                    y[k] = self.Normalize(values[i+train_seq+PH-1],Max=Max,Min=Min)
                    i += train_freq
                    k += 1
                else:
                    i += train_freq
                    k += 1
            return X, y, Max, Min    

 ###################################        access_Excel CLASS        ###################################
class UvaP_dataset:
    def __init__(self, path):
        self.path = path
        self.matrix = self._load_data()
    
    def _load_data(self):
        # Inizializza una lista vuota per accumulare le righe
        rows = []
        # Itera su tutti i file nel percorso specificato
        for filename in os.listdir(self.path):
            if filename.endswith('.xlsx'):
                # Costruisce il percorso completo del file
                file_path = os.path.join(self.path, filename)
                # Legge la prima colonna del file Excel
                df = pd.read_excel(file_path, sheet_name=1, usecols=[3], engine='openpyxl')
                # Trasforma la colonna in una lista e la aggiunge alla lista di righe
                rows.append(df.iloc[1:].values.flatten())
        
        return np.array(rows)

    def get_matrix(self):
        # Restituisce la matrice creata
        return self.matrix

        
 ###################################        access_xml CLASS        ###################################                  
#This class is usefull to manipulate .xml files   
class access_xml:
    def __init__(self,elemento):
        #elemnto must be 'ROOT' of the xml
        self.elemento = elemento
        
    def stampa_elemento(self, input=None, livello=0):
        # This function prints the elements in the xml-file
        if input == None:
            input = self.elemento
        indentazione = '  ' * livello  # This gives indentation for the prints to clarify the level in te xml-file

        # Visualize the elements of the level ('livello')
        print(f"{indentazione}Tag: {input.tag}, Attributi: {input.attrib}")

        # If text is present in the element, print it
        if input.text:
            print(f"{indentazione}Testo: {input.text.strip()}")

        #The function iterates recursively towards deeper and deeper levels
        for figlio in input:
            self.stampa_elemento(figlio, livello + 1)
            
            
    def level_access(self,input=None, livello_desiderato=1, livello_corrente=0):
        #This function reach the desired level in the xml-file and prints its elements 
        if input == None:
            input = self.elemento
        if livello_corrente == livello_desiderato:
        #If the level is the desired level, prints elements
            print("Livello:", livello_corrente)
            print("Tag:", input.tag)
            print("Attributi:", input.attrib)
            print("Testo:", input.text)
        else:
            #If the level is not the desired level, the function iterates recursively towards deeper and deeper levels
            for figlio in input:
                self.level_access(figlio, livello_desiderato, livello_corrente + 1)
        