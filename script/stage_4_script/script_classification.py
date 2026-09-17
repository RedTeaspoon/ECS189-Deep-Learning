import sys, os
# Add project root to path
# Determine project root (two levels up from the script folder)
dir_path = os.path.dirname(__file__)
scripts_dir = os.path.dirname(dir_path)
project_root = os.path.dirname(scripts_dir)
sys.path.append(project_root)

from local_code.stage_4_code.Dataset_TextLoader import Dataset_TextLoader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Setting_Run_RNN import Setting_Run_RNN
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Result_Saver import Result_Saver
from collections import Counter

# --- Dataset loader ---
loader = Dataset_TextLoader(
    dName='TextLoader',
    dDescription='Load IMDB tokenized reviews',
    dataset_root='text_classification'
)
loader.dataset_source_folder_path = os.path.join(
    project_root, 'data', 'stage_4_data'
)

# --- Build vocab from training split ---
data_splits = loader.load()
counter = Counter()
for tokens in data_splits['train']['X']:
    counter.update(tokens)
# initialize vocab with PAD and UNK
default_vocab = {'<PAD>': 0, '<UNK>': 1}
vocab = dict(default_vocab)
for word, freq in counter.items():
    if freq >= 5:
        vocab[word] = len(vocab)

# --- Model ---
model = Method_RNN(
    mName='RNNClassifier',
    mDescription='Vanilla RNN sentiment classifier',
    vocab=vocab,
    embed_dim=100,
    hidden_dim=128,
    num_layers=1,
    num_classes=2,
    max_seq_len=200,
    lr=1e-3,
    batch_size=256,
    epochs=5000
)

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
    project_root, 'result', 'stage_4_result'
)
saver.result_destination_file_name = 'Classification_RNN'

# --- Setting ---
setting = Setting_Run_RNN(
    'RunRNN',
    'Execute RNN training and evaluation'
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