from dataset import dataset
import pickle as pk
from sklearn.model_selection import train_test_split
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Dataset_fake_news_data_large(dataset):
    data = None

    dataset_source_folder_path = None
    dataset_source_file_name = None

    bag = {}


    def __init__(self, dName, dDescription):
        super().__init__(dName, dDescription)

    def load(self):
        #f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        f = open("C:/Users/王博/Desktop/work/1.fake news detection/paper_Jiawei/datasets/fake_news_data_large/dir_webpages/node",
            "rb")
        full_data = pk.load(f)
        data = self.deal_data(full_data) #n*2维列表
        X = [i[0] for i in data]
        X = self.embedding(X)
        #print(X)
        y = [i[1] for i in data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=30, stratify=y)
        print("data_finished!")
        return X_train, X_test, y_train, y_test


    def deal_data(self, full_data):
        data = []
        for i in range(len(full_data) - 13950):
            data.append([])
            for j in range(2):
                data[i].append(0)
        i = 0
        for i in range(len(full_data) - 13950):
            data[i][0] = full_data[i + 1]['content']
            data[i][1] = self.judge_lable(full_data[i + 1]['rating'])

        return data

    def judge_lable(self, rating):
        if rating == 'Mostly False' or rating == 'False' or rating == 'Pants on Fire!':
            return 0
        else:
            return 1


    def embedding(self, X):

        #bag of words
        #vectorizer = CountVectorizer(min_df=1)
        #X = vectorizer.fit_transform(X)

        #tf-idf

        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit_transform(X).todense()
        X = vectorizer.transform(X)

        X = X.toarray()
        return X



'''
if __name__ == '__main__':
    d = Dataset_fake_news_data_large('', '')
    d.load()
'''