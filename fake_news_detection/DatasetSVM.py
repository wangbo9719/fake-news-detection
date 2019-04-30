'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from dataset import dataset
import pickle


class DatasetSVM(dataset):
    data = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    train_test_divide_source_folder_path = None
    article_credibility_dict_folder_path = None
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def reverse_dict(self, input_dict):
        output_dict = {}
        for key in input_dict:
            if not isinstance(input_dict[key], str):
                for value in input_dict[key]:
                    if value not in output_dict:
                        output_dict[value] = {}
                    output_dict[value][key] = 1
            else:
                value = input_dict[key]
                if value not in output_dict:
                    output_dict[value] = {}
                output_dict[value][key] = 1
        return output_dict
    
    def load(self):
        f = open(self.dataset_source_folder_path + 'dir_webpages/node', 'rb')
        article_node_dict = pickle.load(f)
        f.close()
        
        f = open(self.dataset_source_folder_path + 'dir_webpages/link', 'rb')
        article_creator_link_dict = pickle.load(f)
        f.close()
        creator_article_link_dict = self.reverse_dict(article_creator_link_dict)
        
        f = open(self.dataset_source_folder_path + 'source_webpages/creator_node', 'rb')
        creator_node_dict = pickle.load(f)
        f.close()
        
        f = open(self.dataset_source_folder_path + 'subject_webpages/article_index_subject_dict', 'rb')
        article_subject_link_dict = pickle.load(f)
        f.close()
        subject_article_link_dict = self.reverse_dict(article_subject_link_dict)
        
        node_dict = {'creator': creator_node_dict, 'article': article_node_dict}
        link_dict = {'creator_article': creator_article_link_dict, 'article_creator': article_creator_link_dict, 'article_subject': article_subject_link_dict, 'subject_article': subject_article_link_dict}
        
        dataset = {'node': node_dict, 'link': link_dict}
        
        f = open(self.train_test_divide_source_folder_path + 'article_train_test_divide_1.0', 'rb')
        article_train_test_divide_dict = pickle.load(f)
        f.close()
        
        f = open(self.article_credibility_dict_folder_path + 'article_credibility_dic', 'rb')
        article_credibility_dic = pickle.load(f)
        f.close()
        return dataset, article_train_test_divide_dict, article_credibility_dic
    
    