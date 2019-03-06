'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from setting import setting



class SettingTest(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        X_train, X_test, y_train, y_test = self.dataset.load()


        training_set = {'X':X_train, 'y':y_train}
        testing_set = {'X':X_test, 'y':y_test} #.ravel
        
        # run MethodModule
        self.method.data = {'train': training_set, 'test': testing_set}
        learned_result = self.method.run()
        
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
        
        # evaluating ResultModule (optional)
        # evaluation can be done independently after getting the learning ResultModule
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()

        