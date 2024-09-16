import os
import numpy as np
import xml.etree.ElementTree as ET
from utils_dataset import dataset_funcs as df

def save_data_as_npy(current_folder,paz_list_train,paz_list_test):
    if not(os.path.exists(os.path.join(current_folder,'glucose_level.npy'))):

        len_campioni = []
        for paz in range(len(paz_list_train)):
            tree = ET.parse(paz_list_train[paz])
            root = tree.getroot()
            glucose_level = root.find('glucose_level')
            k = 0
            for elemento in glucose_level:
                k += 1
            len_campioni.append(k)

        campioni = np.empty((len(paz_list_train),max(len_campioni)+1,2), dtype=f'U{30}')
            
        for paz in range(len(paz_list_train)):
            i = 0  
            
            #STRUCT DEFINITION
            tree = ET.parse(paz_list_train[paz])
            root = tree.getroot()
                
            #GLUCOSE_LEVEL EXTRACTION
            glucose_level = root.find('glucose_level')
            if glucose_level is not None:
                # Adesso sei all'interno di <glucose_level>. Puoi accedere ai suoi elementi figli:
                for elemento in glucose_level:
                    #print("Attributi:", elemento.attrib['ts'])
                    campioni[paz,i,0] = elemento.attrib['ts']
                    campioni[paz,i,1] = elemento.attrib['value']
                    i += 1
                    
            else:
                print("Elemento <glucose_level> non trovato nel file XML.")
        #Le acquisizioni del livello di glucosio si trovano ora all interno della variabile campioni
        np.save(os.path.join(current_folder,'glucose_level.npy'),campioni)
        

    if not(os.path.exists(os.path.join(current_folder,'glucose_level_test.npy'))):

        len_campioni = []
        for paz in range(len(paz_list_test)):
            tree = ET.parse(paz_list_test[paz])
            root = tree.getroot()
            glucose_level = root.find('glucose_level')
            k = 0
            for elemento in glucose_level:
                k += 1
            len_campioni.append(k)

        campioni = np.empty((len(paz_list_test),max(len_campioni)+1,2), dtype=f'U{30}')
            
        for paz in range(len(paz_list_test)):
            i = 0  
            
            #STRUCT DEFINITION
            tree = ET.parse(paz_list_test[paz])
            root = tree.getroot()
                
            #GLUCOSE_LEVEL EXTRACTION
            glucose_level = root.find('glucose_level')
            if glucose_level is not None:
                # Adesso sei all'interno di <glucose_level>. Puoi accedere ai suoi elementi figli:
                for elemento in glucose_level:
                    #print("Attributi:", elemento.attrib['ts'])
                    campioni[paz,i,0] = elemento.attrib['ts']
                    campioni[paz,i,1] = elemento.attrib['value']
                    i += 1
                    
            else:
                print("Elemento <glucose_level> non trovato nel file XML.")
        #Le acquisizioni del livello di glucosio si trovano ora all interno della variabile campioni
        np.save(os.path.join(current_folder,'glucose_level_test.npy'),campioni)