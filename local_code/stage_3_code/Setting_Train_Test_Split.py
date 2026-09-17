'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    #fold = 3

    """
        Uses the pickled train/test splits provided by Dataset_ImageLoader.
    """

    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train = data['train']['X'], data['train']['y']
        X_test, y_test = data['test']['X'], data['test']['y']
        # feed into method
        self.method.data = {'train': {'X': X_train, 'y': y_train},
                            'test': {'X': X_test, 'y': y_test}}
        result = self.method.run()
        self.result.data = result
        self.result.save()
        self.evaluate.data = result
        acc = self.evaluate.evaluate()
        return acc, None
