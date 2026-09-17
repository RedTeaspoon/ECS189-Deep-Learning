'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np


class Setting_Run_GCN(setting):


    def load_run_save_evaluate(self):

        data = self.dataset.load()

        # run MethodModule
        self.method.data = data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()
        if isinstance(metrics, dict):
            accuracy = metrics.get('accuracy', metrics.get('accuracy', metrics))
        else:
            accuracy = metrics
        #accuracy = metrics['accuracy']

        return np.mean([accuracy]), np.std([accuracy])


