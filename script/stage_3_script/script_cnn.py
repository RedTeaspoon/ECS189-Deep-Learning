import sys, os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from local_code.stage_3_code.Dataset_ImageLoader import Dataset_ImageLoader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
#-----------Choosing Device---------------
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device:", device)
#-----------------------------------------
#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # Choose dataset: 'ORL', 'MNIST', or 'CIFAR'
    dataset_name = 'ORL'
    file_map = {'MNIST': 'MNIST', 'ORL': 'ORL', 'CIFAR': 'CIFAR'}

    # ---- objection initialization setction ---------------
    #data_obj = Dataset_ImageLoader('stage3', dataset_name, file_map[dataset_name])
    #data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    loader = Dataset_ImageLoader(
        dName='ImageLoader',
        dDescription=f'Load {dataset_name}',
        file_name=file_map[dataset_name]
    )
    loader.dataset_source_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'stage_3_data') + os.sep

    # Model parameters
    if dataset_name == 'ORL':
        in_ch, img_sz, num_cls = 1, (112, 92), 40
    elif dataset_name == 'MNIST':
        in_ch, img_sz, num_cls = 1, 28, 10
    else:  # CIFAR
        in_ch, img_sz, num_cls = 3, 32, 10

    model = Method_CNN(
        mName='SimpleCNN',
        mDescription=f'CNN for {dataset_name}',
        in_channels=in_ch,
        num_classes=num_cls,
        img_size=img_sz
    )
    model.to(device)
    # --- Pipeline components ---
    setting = Setting_Train_Test_Split('Split', 'Use provided train/test')
    saver = Result_Saver('Saver', 'Save results')
    saver.result_destination_folder_path = os.path.join(grandparent_dir, 'result', 'stage_3_result') + os.sep
    saver.result_destination_file_name = f'{dataset_name}_CNN'
    evaluator = Evaluate_Accuracy('Accuracy', 'Compute accuracy')

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting.prepare(loader, model, saver, evaluator)
    setting.print_setup_summary()
    mean_acc, std_acc = setting.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print(f"Final Test Accuracy: {mean_acc:.4f}")
    print('************ Finish ************')
    # ------------------------------------------------------




    