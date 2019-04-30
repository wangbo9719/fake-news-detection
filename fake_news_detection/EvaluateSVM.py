'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from evaluate import evaluate
from sklearn import metrics
import numpy as np

class Evaluate_SVM(evaluate):
    data = None
    
    def evaluate_in_npratio (self, all_fold_results):
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        for fold_num in all_fold_results:
            y_true = all_fold_results[fold_num][0]   #y_true是列表
            y_pred = all_fold_results[fold_num][1]

            accuracy = metrics.accuracy_score(y_true,y_pred)
            f1 = metrics.f1_score(y_true, y_pred, average = 'binary')
            precision = metrics.precision_score(y_true, y_pred, average = 'binary')
            recall = metrics.recall_score(y_true, y_pred, average = 'binary')
            
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)
        accuracy_mean = np.mean(accuracy_list)
        accuracy_std = np.std(accuracy_list) #标准差
        f1_mean = np.mean(f1_list)
        f1_std = np.std(f1_list)
        precision_mean = np.mean(precision_list)
        precision_std = np.std(precision_list)
        recall_mean = np.mean(recall_list)
        recall_std = np.std(recall_list)
        return {'accuracy_mean': accuracy_mean, 'accuracy_std' : accuracy_std, 'f1_mean': f1_mean, 'f1_std' : f1_std, 'precision_mean': precision_mean, 'precision_std': precision_std, 'recall_mean': recall_mean, 'recall_std': recall_std}
    
    def evaluate(self, svm_result):
        evaluate_result = {}
        evaluate_result['mp+md'] = {}
        for np_ratio in svm_result['mp+md']:
            npratio_result = self.evaluate_in_npratio(svm_result['mp+md'][np_ratio])
            evaluate_result['mp+md'][np_ratio] = npratio_result
        evaluate_result['md'] = {}
        for np_ratio in svm_result['md']:
            npratio_result = self.evaluate_in_npratio(svm_result['md'][np_ratio])
            evaluate_result['md'][np_ratio] = npratio_result
        evaluate_result['mp'] = {}
        for np_ratio in svm_result['mp']:
            npratio_result = self.evaluate_in_npratio(svm_result['mp'][np_ratio])
            evaluate_result['mp'][np_ratio] = npratio_result
        evaluate_result['time'] = {}
        evaluate_result['time'] = svm_result['time']
        return evaluate_result