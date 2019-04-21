import pickle
import pandas as pd

if 1:
    f = open(
        "C:/Users/王博/Desktop/work/1.fake news detection/paper_Jiawei/datasets/fake_news_data_large/dir_webpages/node",
        "rb")
    dict = pickle.load(f)
    pos_examples = []
    neg_examples = []
    for i in range(1, len(dict)):
        if dict[i]['content'] != '':
            if dict[i]['rating'] == 'Half_True' or dict[i]['rating'] == 'Mostly True' or dict[i]['rating'] == 'True':
                pos_examples.append([dict[i]['content'],0])
            else:
                neg_examples.append([dict[i]['content'],1])
    names = ['content','rating']
    pos_examples = pd.DataFrame(columns=names, data = pos_examples)
    neg_examples = pd.DataFrame(columns=names, data = neg_examples)
    pos_examples.to_csv('./data_samples/pos_examples.csv')
    neg_examples.to_csv('./data_samples/neg_examples.csv')