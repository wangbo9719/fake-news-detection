from dataset import dataset
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data  # 导入手写数字数据集


class Dataset_Mnist(dataset):
    data = None

    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName, dDescription):
        super().__init__(dName, dDescription)

    def load(self):
        data = input_data.read_data_sets('../../MNIST_data', one_hot=True)
        return data