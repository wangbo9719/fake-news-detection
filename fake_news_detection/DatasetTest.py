'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from dataset import dataset


class DatasetTest(dataset):
    data = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName, dDescription):
        super().__init__(dName, dDescription)
    
    def load(self):
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name)
        for line in f:
            line = line.strip('\n')
            [feature, label] = line.split('\t')
            X.append([float(feature)])
            y.append(int(label))
        return {'X': X, 'y': y}