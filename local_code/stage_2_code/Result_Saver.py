'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
import os
import matplotlib.pyplot as plt

class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        # pickle the full result dict
        pkl_path = os.path.join(
            self.result_destination_folder_path,
            f"{self.result_destination_file_name}.pkl"
        )
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Saved raw results to {pkl_path}")

        if 'epochs' in self.data:
            eps = self.data['epochs']
            train_loss = self.data['train_loss']
            test_loss = self.data['test_loss']

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(eps, train_loss, label='Train Loss', color='steelblue', linewidth=2)
            ax.plot(eps, test_loss, label='Test Loss', color='salmon',     linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            img_path = os.path.join(
                self.result_destination_folder_path,
                f"{self.result_destination_file_name}_learning_curve.png"
            )
            fig.savefig(img_path, dpi=300, bbox_inches='tight' )
            plt.close(fig)
            print(f"Saved learning-curve plot to {img_path}")

            eps = self.data['epochs']
            train_acc = self.data['train_acc']
            test_acc = self.data['test_acc']

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(eps, train_acc, label='Train Accuracy', color='steelblue', linewidth=2)
            ax.plot(eps, test_acc, label='Test Accuracy', color='salmon',     linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curve')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            img_path = os.path.join(
                self.result_destination_folder_path,
                f"{self.result_destination_file_name}_accuracy_curve.png"
            )
            fig.savefig(img_path, dpi=300, bbox_inches='tight' )
            plt.close(fig)
            print(f"Saved accuracy-curve plot to {img_path}")