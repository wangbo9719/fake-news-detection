'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
import random
#from sklearn.model_selection import KFold

class MethodSampleCV(method):
    alpha = 0.0
    data = None
    train_ratio = 0
    sample_ratios = None
    def extract_article_train_test_index_list(self):
        train_index_list = []
        test_index_list = []
        article_node_dic = self.data['node']['article']
        index_list = list(range(1,len(article_node_dic)+1))
        train_index_list = random.sample(index_list,int(len(index_list)*self.train_ratio))
        test_index_list = [i for i in index_list if i not in train_index_list]
        
        return train_index_list,test_index_list
    
    def sample_from_train_index(self, sample_ratio, train_index_list):
        sampled_train_index_list = random.sample(train_index_list,int(len(train_index_list)*sample_ratio))
        return sampled_train_index_list
    
    def run(self):
        article_train_test_index_dic = {}
        article_train_index_list, article_test_index_list = self.extract_article_train_test_index_list()
        for ratio in self.sample_ratios:
          article_train_test_index_dic[ratio] = {}
          article_train_test_index_dic[ratio]['test'] = article_test_index_list
          article_train_test_index_dic[ratio]['train'] = self.sample_from_train_index(ratio,article_train_index_list)
        return article_train_test_index_dic