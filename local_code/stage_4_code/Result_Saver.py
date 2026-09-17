'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle, os
import matplotlib.pyplot as plt

class Result_Saver(result):

    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        # ensure destination directory exists
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        # 1) pickle
        pkl_path = os.path.join(
            self.result_destination_folder_path,
            f"{self.result_destination_file_name}.pkl"
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Saved raw results to {pkl_path}")

        # 2) plot if we have epochs
        if 'epochs' in self.data:
            eps = self.data['epochs']
            train_loss = self.data['train_loss']
            test_loss = self.data['test_loss']
            train_acc = self.data['train_acc']
            test_acc = self.data['test_acc']
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(eps, train_loss, label='Train Loss')
            ax.plot(eps, test_loss, label='Test Loss')
            ax.set_xlabel('Epoch');
            ax.set_ylabel('Loss')
            ax.legend();
            ax.grid(True)
            loss_path = os.path.join(
                self.result_destination_folder_path,
                f"{self.result_destination_file_name}_loss_curve.png"
            )
            fig.savefig(loss_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved loss‐curve to {loss_path}")

            # Accuracy curve
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(eps, train_acc, label='Train Acc')
            ax.plot(eps, test_acc, label='Test Acc')
            ax.set_xlabel('Epoch');
            ax.set_ylabel('Accuracy')
            ax.legend();
            ax.grid(True)
            acc_path = os.path.join(
                self.result_destination_folder_path,
                f"{self.result_destination_file_name}_acc_curve.png"
            )
            fig.savefig(acc_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved accuracy‐curve to {acc_path}")
