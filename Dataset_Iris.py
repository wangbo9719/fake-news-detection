'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from dataset import dataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Dataset_Iris(dataset):

    data = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName, dDescription):
        super().__init__(dName, dDescription)
    
    def load(self):
        f = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        data = np.array(f)
        X_train, X_test, y_train, y_test = train_test_split(data[:, :4], data[:, 4:], test_size=0.3,
                                                            random_state=30, stratify=data[:,4:])
        return X_train, X_test, y_train, y_test
        #return {'X': data[:,:4], 'y': data[:,4:]}


