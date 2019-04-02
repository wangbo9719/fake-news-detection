'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from evaluate import evaluate


class EvaluateTest(evaluate):
    data = None

    def evaluate(self):
        num = 0
        for i, j in zip(self.data['pred_y'], self.data['true_y']):
            if i == j:
                num += 1
        acc = num/len(self.data['pred_y'])
        return acc