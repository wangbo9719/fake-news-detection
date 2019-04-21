from dataset import dataset
import pickle
import pandas as pd
#from debian.debtags import output

class DatasetPolitiFact(dataset):
    data = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
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

        return {'node': node_dict, 'link': link_dict}
'''
        pos_examples = []
        neg_examples = []
        for i in range(1, len(article_node_dict)):
            if article_node_dict[i]['content'] != '':
                if article_node_dict[i]['rating'] == 'Half_True' or article_node_dict[i]['rating'] == 'Mostly True' or article_node_dict[i][
                    'rating'] == 'True':
                    pos_examples.append([article_node_dict[i]['content'], 0])
                else:
                    neg_examples.append([article_node_dict[i]['content'], 1])
        names = ['content', 'rating']
        pos_examples = pd.DataFrame(columns=names, data=pos_examples)
        neg_examples = pd.DataFrame(columns=names, data=neg_examples)
        return {'pos': pos_examples, 'neg': neg_examples}
'''


