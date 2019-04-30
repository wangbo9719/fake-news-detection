'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from setting import setting


class SettingSampleCV(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        

        # run MethodModule
        self.method.data = loaded_data
        divide_result = self.method.run()
        
        # save raw ResultModule
        self.result.data = divide_result
        self.result.save()
        
        # evaluating ResultModule (optional)
        # evaluation can be done independently after getting the learning ResultModule
#        self.evaluate.data = learned_result
        return divide_result

        