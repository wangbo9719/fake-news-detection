'''
Base MethodModule class for all models and frameworks
'''



from DatasetPolitiFact import DatasetPolitiFact
import pickle
import json
class CreateCredibilityDict:
    alpha = 1.0
    data_type = '_large'

    
    def load_data(self):
        data_obj = DatasetPolitiFact('', '')
        data_obj.dataset_source_folder_path = './fake_news_data' + self.data_type + '/'
        loaded_data = data_obj.load()
        return loaded_data  #node_dict = {'creator': creator_node_dict, 'article': article_node_dict}
    def create_article_credibility_dic(self,loaded_data):
        article_node_dic = loaded_data['node']['article']
        article_credibility_dic = {}
        for key in article_node_dic:
            if article_node_dic[key]['rating'] == 'Pants on Fire!':
                article_credibility_dic[key] = 1
            elif article_node_dic[key]['rating'] == 'False':
                article_credibility_dic[key] = 2
            elif article_node_dic[key]['rating'] == 'Mostly False':
                article_credibility_dic[key] = 3
            elif article_node_dic[key]['rating'] == 'Half-True':
                article_credibility_dic[key] = 4
            elif article_node_dic[key]['rating'] == 'Mostly True':
                article_credibility_dic[key] = 5
            elif article_node_dic[key]['rating'] == 'True':
                article_credibility_dic[key] = 6
            else:
                print(key)
                print(article_node_dic[key]['rating'])
        return article_credibility_dic
    
if 1:
    create = CreateCredibilityDict()
    loaded_data = create.load_data()
    article_credibility_dic = create.create_article_credibility_dic(loaded_data)

    #article_credibility_dic
    #jsoninfo = json.dumps(article_credibility_dic)
    #print(jsoninfo)

    f = open('./data_samples/article_credibility_dic', 'wb')
    pickle.dump(article_credibility_dic, f)
    f.close()
    
        
