'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from result import result
import pickle


class ResultSampleCV(result):
    data = None
    
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()