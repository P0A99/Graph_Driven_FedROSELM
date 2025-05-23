import pandas as pd
import os
import numpy as np
def sim_martix():
    current_dir = os.path.dirname(__file__)


    excel_path = os.path.join(current_dir, '../Dati/Similarity_matrix_Ohio.csv')
    df = pd.read_csv(excel_path)

    cluster1 = np.array(df.iloc[[0,1,2,4,5,8,11], [1,2,3,5,6,9,12]].values)
    cluster2 = np.array(df.iloc[[3,6,9,10], [4,7,10,11]].values)

    matrix = np.array(df.iloc[:,1:])


    print(matrix)
    return matrix