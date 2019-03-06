from Dataset_Iris import Dataset_Iris
from Dataset_fake_news_data_large import Dataset_fake_news_data_large
from Method_svm import Method_svm
from ResultTest import ResultTest
from SettingTest import SettingTest
from EvaluateTest import EvaluateTest
import pickle


if 1:
    #---- parameter section -------------------------------
    alpha = 1.0
    #------------------------------------------------------
    
    #---- objection initialization setction ---------------
    data_obj = Dataset_fake_news_data_large('', '')
    #data_obj.dataset_source_folder_path = './data_samples/'
    #data_obj.dataset_source_file_name = 'iris.csv'
    
    method_obj = Method_svm('', '')
    method_obj.alpha = alpha
    
    result_obj = ResultTest('', '')
    #result_obj.result_destination_folder_path = './result_samples/'
    #result_obj.result_destination_file_name = 'prediction_result' + str(alpha)
    result_obj.result_destination_folder_path = './result_samples/'
    result_obj.result_destination_file_name = 'fake_news_result'
    
    setting_obj = SettingTest('', '')
    
    evaluate_obj = EvaluateTest('', '')
    #------------------------------------------------------
    
    #---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    evaluation_result = setting_obj.load_run_save_evaluate()
    print(evaluation_result)

    f = open('./result_samples/fake_news_result', 'rb')
    #f = open('./result_samples/prediction_result_1.0','rb')
    d = pickle.load(f)
    print(f)
    #------------------------------------------------------
    