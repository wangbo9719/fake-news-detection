import pickle
import json

f = open("C:/Users/王博/Desktop/work/1.fake news detection/paper_Jiawei/datasets/fake_news_data_large/dir_webpages/node","rb")
d = pickle.load(f)
f2 = open("C:/Users/王博/Desktop/node.txt","w")
jsoninfo = json.dumps(d) # 输出str类型
f2.write(jsoninfo)
f2.close()