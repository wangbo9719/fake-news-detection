'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from setting import setting


class SettingSVM(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        dataset, article_train_test_divide_dict, article_credibility_dic = self.dataset.load()
        

        # run MethodModule
        self.method.data = dataset
        self.method.train_test_divide = article_train_test_divide_dict
        self.method.article_credibility_dict = article_credibility_dic
        
        svm_result = self.method.run()
        

        self.result.data = svm_result
        self.result.save()
        
        # evaluating ResultModule (optional)
        # evaluation can be done independently after getting the learning ResultModule
#        self.evaluate.data = learned_result
        return svm_result

        