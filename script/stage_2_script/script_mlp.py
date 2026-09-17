import sys, os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Run_MLP import Setting_Run_MLP
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage2', '784 features')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name_train = 'train.csv'
    data_obj.dataset_source_file_name_test = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Run_MLP('Setting run MLP', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', 'Check the accuracy, f1, recall, and precision of the predicted y')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    test_eval = setting_obj.load_run_save_evaluate()

    print('************ Final Test Performance ************')
    print('MLP Accuracy: ', test_eval['accuracy'])
    print('MLP F1 (Macro / Micro / Weighted): ', test_eval['f1_macro'], ' / ', test_eval['f1_micro'], ' / ', test_eval['f1_weighted'])
    print('MLP Precision (Macro / Micro / Weighted): ', test_eval['precision_macro'], ' / ', test_eval['precision_micro'], ' / ', test_eval['precision_weighted'])
    print('MLP Recall (Macro / Micro / Weighted): ', test_eval['recall_macro'], ' / ', test_eval['recall_micro'], ' / ', test_eval['recall_weighted'])
    print('************ Finish ************')
    # ------------------------------------------------------
    

    