from DatasetPolitiFact import DatasetPolitiFact
from MethodTextCNN import MethodTextCNN
from ResultTest import ResultTest
from SettingCV import SettingCV
from EvaluateAcc import EvaluateAcc
from DatasetSVM import DatasetSVM

from MethodSVM import MethodSVM
from ResultTest import ResultTest
from ResultSampleCV import ResultSampleCV
from SettingTest import SettingTest
from SettingSampleCV import SettingSampleCV
from SettingSVM import SettingSVM
from EvaluateTest import EvaluateTest
from SettingTextCNN import SettingTextCNN
from MethodAdversarial import MethodAdversarial
from SettingAdversarial import SettingAdversarial
import pickle

# ---- Adversarial ----
if 1:
    # ---- parameter section -------------------------------
    alpha = 1.0
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = DatasetSVM('', '')
    # data_obj.dataset_source_folder_path = '../../data/fake_news_data' + data_type + '/'
    data_obj.dataset_source_folder_path = 'D:/fake_news_detection/fake_news_data_large/'
    data_obj.train_test_divide_source_folder_path = './data_samples/'
    data_obj.article_credibility_dict_folder_path = './data_samples/'

    method_obj = MethodAdversarial('','')
    method_obj.alpha = alpha
    method_obj.sample_ratios = [1.0]

    result_obj = ResultTest('', '')
    result_obj.result_destination_folder_path = './result_samples/'
    result_obj.result_destination_file_name = 'Adversarial' + str(alpha)

    evaluate_obj = EvaluateAcc('', '')

    setting_obj = SettingAdversarial('', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    evaluation_result = setting_obj.load_run_save_evaluate()
    print(evaluation_result)
    # ------------------------------------------------------

#---- text_CNN ----
if 0:
    #---- parameter section -------------------------------
    alpha = 1.0
    #------------------------------------------------------
    
    #---- objection initialization setction ---------------
    data_obj = DatasetSVM('', '')
    # data_obj.dataset_source_folder_path = '../../data/fake_news_data' + data_type + '/'
    data_obj.dataset_source_folder_path = 'D:/fake_news_detection/fake_news_data_large/'
    data_obj.train_test_divide_source_folder_path = './data_samples/'
    data_obj.article_credibility_dict_folder_path = './data_samples/'

    
    method_obj = MethodTextCNN('', '')
    method_obj.alpha = alpha
    method_obj.sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    
    result_obj = ResultTest('', '')
    result_obj.result_destination_folder_path = './result_samples/'
    result_obj.result_destination_file_name = 'textCNN' + str(alpha)
    
    evaluate_obj = EvaluateAcc('', '')

    setting_obj = SettingTextCNN('', '')
    #------------------------------------------------------
    
    #---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    evaluation_result = setting_obj.load_run_save_evaluate()
    print(evaluation_result)
    #------------------------------------------------------

# -------------------------SVM---------------------
if 0:
    # ---- parameter section -------------------------------
    alpha = 1.0
    data_type = '_large'
    c = 1.0
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = DatasetSVM('', '')
    # data_obj.dataset_source_folder_path = '../../data/fake_news_data' + data_type + '/'
    data_obj.dataset_source_folder_path = 'D:/fake_news_detection/fake_news_data_large/'
    data_obj.train_test_divide_source_folder_path = './data_samples/'
    data_obj.article_credibility_dict_folder_path = './data_samples/'

    method_obj = MethodSVM('', '')
    method_obj.alpha = alpha
    method_obj.c = c
    method_obj.sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #    method_obj.sample_ratios = [1.0]
    result_obj = ResultTest('', '')
    result_obj.result_destination_folder_path = './result_samples/'
    result_obj.result_destination_file_name = 'multi_class_SVM_with_ratio' + str(alpha)

    setting_obj = SettingSVM('', '')

    evaluate_obj = EvaluateTest('', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    divide_result = setting_obj.load_run_save_evaluate()
    print(divide_result)
    # ------------------------------------------------------
    