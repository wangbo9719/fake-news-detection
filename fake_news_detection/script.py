from DatasetPolitiFact import DatasetPolitiFact
from MethodTextCNN import MethodTextCNN
from ResultTest import ResultTest
from SettingCV import SettingCV
from EvaluateAcc import EvaluateAcc
#---- text_CNN ----
if 1:
    #---- parameter section -------------------------------
    alpha = 1.0
    #------------------------------------------------------
    
    #---- objection initialization setction ---------------
    data_obj = DatasetPolitiFact('', '')
    data_obj.dataset_source_folder_path = './fake_news_data_large/'

    
    method_obj = MethodTextCNN('', '')
    method_obj.alpha = alpha
    
    result_obj = ResultTest('', '')
    result_obj.result_destination_folder_path = './result_samples/'
    result_obj.result_destination_file_name = 'textCNN' + str(alpha)
    
    evaluate_obj = EvaluateAcc('', '')

    setting_obj = SettingCV('', '')
    #------------------------------------------------------
    
    #---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    evaluation_result = setting_obj.load_run_save_evaluate()
    print(evaluation_result)
    #------------------------------------------------------
    