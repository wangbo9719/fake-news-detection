'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
from sklearn import svm
import collections
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import string
from sklearn.feature_extraction.text import TfidfVectorizer

class MethodSVM(method):
    c = None
    data = None
    train_test_divide = None
    article_credibility_dict = None
    sample_ratios = None
    
    article_train_index_list = None
    article_test_index_list = None
    
    def get_word_dict(self):
        word_list = []
        self.max_string_length = 0
        length_list = []
        for article in self.data['node']['article']:
            content = self.data['node']['article'][article]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            #把content中的符号替换为空，并换成小写后，按空格分成word列表
            words = content.lower().split(' ')
            
            if len(words) > 100: #words最长只要100个词？
                words = words[:100]
            word_list.extend(words)
            if len(words) > self.max_string_length:
                self.max_string_length = len(words)
            length_list.append(len(words))

        word_list.append('dummy_word_that_will_never_appear')
        return word_list
    
    def build_dataset(self, sample_ratio, word_list):       
        self.article_train_index_list = self.train_test_divide[sample_ratio]['train']
        self.article_test_index_list = self.train_test_divide[sample_ratio]['test']
        count = collections.Counter(word_list).most_common()
        #Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value
        #most_common()返回topN个，为空则全部返回
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.vocab_size = len(self.dictionary)
        self.padding_x = self.dictionary['dummy_word_that_will_never_appear']
    
    def batch_generation(self):
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: 
                self.article_train_index_list.remove(article_train_index)
        article_train_X = np.matrix([[self.padding_x]*self.max_string_length]*len(self.article_train_index_list))
        article_train_y = np.array([0.0]*len(self.article_train_index_list))
        order_num_train = 0 
        
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_train_index]
            article_train_y[order_num_train] = y
            content = self.data['node']['article'][article_train_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            if len(words_list) > 100:
                words_list = words_list[:100]
            for word_index in range(len(words_list)):
                value = self.dictionary[words_list[word_index]]
                article_train_X[order_num_train, word_index] = value
            order_num_train += 1
            
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: 
#                print(article_test_index)
                self.article_test_index_list.remove(article_test_index)
        article_test_X = np.matrix([[self.padding_x]*self.max_string_length]*len(self.article_test_index_list))
        article_test_y = np.array([0.0]*len(self.article_test_index_list))
        order_num_test = 0 
        
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_test_index]
            article_test_y[order_num_test] = y
#            words_list = self.data['node']['article'][article_test_index]['content'].lower().split(' ')
            content = self.data['node']['article'][article_test_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            if len(words_list) > 100:
                words_list = words_list[:100]
            for word_index in range(len(words_list)):
                value = self.dictionary[words_list[word_index]]
                article_test_X[order_num_test, word_index] = value
            order_num_test += 1
#        print(order_num_test+order_num_train)
        return article_train_X, article_train_y, article_test_X, article_test_y
    
    def bi_class_batch_generation(self):
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: 
                self.article_train_index_list.remove(article_train_index)
        article_train_X = np.matrix([[self.padding_x]*self.max_string_length]*len(self.article_train_index_list))
        article_train_y = np.array([0.0]*len(self.article_train_index_list))
        order_num_train = 0 
        
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_train_index]
            if y>=4:
                y = 1
            else:
                y =0
            article_train_y[order_num_train] = y
            content = self.data['node']['article'][article_train_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            if len(words_list) > 100:
                words_list = words_list[:100]
            for word_index in range(len(words_list)):
                value = self.dictionary[words_list[word_index]]
                article_train_X[order_num_train, word_index] = value
            order_num_train += 1
            
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: 
                self.article_test_index_list.remove(article_test_index)
        article_test_X = np.matrix([[self.padding_x]*self.max_string_length]*len(self.article_test_index_list))
        article_test_y = np.array([0.0]*len(self.article_test_index_list))
        order_num_test = 0 
        
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_test_index]
            if y>=4:
                y = 1
            else:
                y =0
            article_test_y[order_num_test] = y
            content = self.data['node']['article'][article_test_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            if len(words_list) > 100:
                words_list = words_list[:100]
            for word_index in range(len(words_list)):
                value = self.dictionary[words_list[word_index]]
                article_test_X[order_num_test, word_index] = value
            order_num_test += 1
        return article_train_X, article_train_y, article_test_X, article_test_y    
    
    def bow_bi_class_batch_generation(self):

        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: 
                self.article_train_index_list.remove(article_train_index)
        article_train_X = np.matrix([[0]*len(self.dictionary)]*len(self.article_train_index_list))
        article_train_y = np.array([0.0]*len(self.article_train_index_list))
        order_num_train = 0 
        
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_train_index]
#            if y>=4:
#                y = 1
#            else:
#                y =0
            article_train_y[order_num_train] = y
            content = self.data['node']['article'][article_train_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            count = collections.Counter(words_list).most_common()
            for word, num in count:
                element_index = int(self.dictionary[word])
                article_train_X[order_num_train, element_index] = num
            order_num_train += 1
            
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: 
                self.article_test_index_list.remove(article_test_index)
        article_test_X = np.matrix([[self.padding_x]*len(self.dictionary)]*len(self.article_test_index_list))
        article_test_y = np.array([0.0]*len(self.article_test_index_list))
        order_num_test = 0 
        
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_test_index]
#            if y>=4:
#                y = 1
#            else:
#                y =0
            article_test_y[order_num_test] = y
            content = self.data['node']['article'][article_test_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            words_list = content.lower().split(' ')
            count = collections.Counter(words_list).most_common()
            for word, num in count:
                element_index = int(self.dictionary[word])
                article_test_X[order_num_test, element_index] = num
            order_num_test += 1
        return article_train_X, article_train_y, article_test_X, article_test_y   
    
    def tfidf_bi_class_batch_generation(self):
        
        corpus = []
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: 
                self.article_train_index_list.remove(article_train_index)

        article_train_y = np.array([0.0]*len(self.article_train_index_list))
        order_num_train = 0 
        
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_train_index]
            #二分类处理
            if y>=4:
                y = 1
            else:
                y =0
            article_train_y[order_num_train] = y
            content = self.data['node']['article'][article_train_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            corpus.append(content)
            order_num_train += 1
        tfidf = TfidfVectorizer()


        #--------test_data------------
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: 
                self.article_test_index_list.remove(article_test_index)

        article_test_y = np.array([0.0]*len(self.article_test_index_list))
        order_num_test = 0 
        
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_test_index]
            if y>=4:
                y = 1
            else:
                y =0
            article_test_y[order_num_test] = y
            content = self.data['node']['article'][article_test_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            corpus.append(content)
            order_num_test += 1
        #----------------------
        #order_num_train和order_num_test用来方便对形成的tf-idf矩阵进行分割
        article_train_y = article_train_y[:order_num_train] 
        re = tfidf.fit_transform(corpus).todense()
        article_train_X = re[:order_num_train,:] 
        article_test_X = re[order_num_train:,:]
        return article_train_X, article_train_y, article_test_X, article_test_y   

    def train(self, X, y):
        min_max_scaler = preprocessing.MinMaxScaler() #归一化
        model = svm.LinearSVC(C = self.c, multi_class='ovr')
        model.fit(min_max_scaler.fit_transform(X), y)
        #trainy一个svm分类器，返回它以及训练数据x归一化后的结果
        return model, min_max_scaler

    #用训练好的svm分类器，对测试数据进行同样的归一化后，预测结果
    def test(self, model, min_max_scaler, X):
        return model.predict(min_max_scaler.transform(X))
    
    def run(self):
        word_list = self.get_word_dict()
        result = {}
        for ratio in self.sample_ratios: #sample_rations一个列表
            self.build_dataset(ratio, word_list)
            article_train_X, article_train_y, article_test_X, article_test_y = self.tfidf_bi_class_batch_generation()
            print('ratio:',ratio)
            '''
            print('article_train_X.shape:',article_train_X.shape)
            print('article_train_y.shape:',article_train_y.shape)
            print('article_test_X.shape:',article_test_X.shape)
            print('article_test_y.shape:',article_test_y.shape)
            '''
            #---- article training ----
            print('Article Model Training and Testing.')
            article_model, min_max_scaler = self.train(article_train_X, article_train_y)
            article_pred_y = self.test(article_model, min_max_scaler, article_test_X)
#            article_pred_y = article_model.predict(article_test_X)
#            print(article_pred_y)
#            print(article_test_y)
            print(accuracy_score(article_test_y, article_pred_y))
            article_result = {'pred_y': article_pred_y, 'true_y': article_test_y}
            result[ratio] = article_result
        return result