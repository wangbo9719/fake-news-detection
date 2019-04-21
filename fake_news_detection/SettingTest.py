'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from setting import setting


class SettingTest(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        # training, testing set partition
        training_set = {'X': [loaded_data['X'][i] for i in [0, 1, 2, 5, 6, 7]], 
                        'y': [loaded_data['y'][i] for i in [0, 1, 2, 3, 5, 6, 7]]}
        testing_set = {'X': [loaded_data['X'][i] for i in [3, 4, 8, 9]], 
                       'y': [loaded_data['y'][i] for i in [3, 4, 8, 9]]}
        
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

        