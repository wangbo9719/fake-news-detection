'''
Base MethodModule class for all models and frameworks
'''



from DatasetPolitiFact import DatasetPolitiFact
import pickle
import json
import re
class CreateContentDict:
    alpha = 1.0
    data_type = '_large'

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def load_data(self):
        data_obj = DatasetPolitiFact('', '')
        data_obj.dataset_source_folder_path = './fake_news_data' + self.data_type + '/'
        loaded_data = data_obj.load()
        return loaded_data  #node_dict = {'creator': creator_node_dict, 'article': article_node_dict}
    def create_article_content_dic(self,loaded_data):
        article_node_dic = loaded_data['node']['article']
        article_content_dic = {}
        for key in article_node_dic:
            if article_node_dic[key]['rating'] != '':
                content = self.clean_str(article_node_dic[key]['content'])
                article_content_dic[key] = content
            else:
                print(key)
                print(article_node_dic[key]['content'])
        return article_content_dic
    
if 1:
    create = CreateContentDict()
    loaded_data = create.load_data()
    article_content_dic = create.create_article_content_dic(loaded_data)
    f = open('./data_samples/article_content_dic', 'wb')
    pickle.dump(article_content_dic, f)
    f.close()
    
        
