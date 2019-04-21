'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method


class MethodTest(method):
    alpha = 0.0
    data = None
    
    def train(self, training_set):
        pass
    
    def run(self):
        model = self.train(self.data['train'])
        pred_y = []
        true_y = []
        for x, y in zip(self.data['test']['X'], self.data['test']['y']):
            pred_y.append(x[0] <= self.alpha and 1 or 0)
            true_y.append(y)
        return {'pred_y': pred_y, 'true_y': true_y}
            