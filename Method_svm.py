'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np


class Method_svm(method):
    alpha = 0.4
    data = None

    def train(self, training_X, training_y):
        parameters = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': np.linspace(0.1, 20, 50),
                      'gamma': np.linspace(0.1, 20, 20)}
        svc = svm.SVC()
        model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
        model.fit(training_X, training_y)
        print("train finished!")
        return model
    
    def run(self):
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        pred_y = model.predict(self.data['test']['X'])
        print(pred_y,self.data['test']['y'])
        print("predict finished!")
        return {'pred_y': pred_y, 'true_y':self.data['test']['y']}
            