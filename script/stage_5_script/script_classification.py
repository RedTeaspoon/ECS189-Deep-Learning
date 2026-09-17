import sys, os
# Add project root to path
# Determine project root (two levels up from the script folder)
dir_path = os.path.dirname(__file__)
scripts_dir = os.path.dirname(dir_path)
project_root = os.path.dirname(scripts_dir)
sys.path.append(project_root)

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Setting_Run_GCN import Setting_Run_GCN
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Result_Saver import Result_Saver
import numpy as np
import torch

#---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
#------------------------------------------------------

# --- Dataset loader ---
loader = Dataset_Loader(
    dName='DataLoader',
    dDescription='Load '
)
# Choose dataset: 'cora', 'citeseer', or 'pubmed'
loader.dataset_name = 'pubmed'
loader.dataset_source_folder_path = os.path.join(
    project_root, 'data', 'stage_5_data', loader.dataset_name
)

if loader.dataset_name == 'cora':
    num_feat, num_cls = 1433, 7
elif loader.dataset_name == 'citeseer':
    num_feat, num_cls = 3703, 6
else:  # pubmed
    num_feat, num_cls = 500, 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model ---
model = Method_GCN(
    mName='GCNClassifier',
    mDescription='GCN node classifier',
    num_features=num_feat,
    num_classes=num_cls,
    #lr=1e-3,
    epochs=500, 
    patience=10
)
model.to(device)

# --- Evaluator and Saver ---
evaluator = Evaluate_Accuracy(
    'Accuracy',
    'Compute classification accuracy'
)
saver = Result_Saver(
    'Saver',
    'Save outputs'
)
saver.result_destination_folder_path = os.path.join(
    project_root, 'result', 'stage_5_result'
)
saver.result_destination_file_name = f'{loader.dataset_name}_GCN'

# --- Setting ---
setting = Setting_Run_GCN(
    'RunGCN',
    'Execute GCN training and evaluation'
)

# ---- running section ---------------------------------
print('************ Start ************')
setting.prepare(loader, model, saver, evaluator)
setting.print_setup_summary()
mean_acc, std_acc = setting.load_run_save_evaluate()

print('************ Overall Performance ************')
print(f"Final Test Accuracy: {mean_acc:.4f}")
print('************ Finish ************')
# ------------------------------------------------------