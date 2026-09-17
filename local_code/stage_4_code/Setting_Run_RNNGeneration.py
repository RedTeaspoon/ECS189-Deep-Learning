'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np


class Setting_Run_RNNGeneration(setting):


    def load_run_save_evaluate(self):

        # 1) load jokes dataset
        data_full = self.dataset.load()  # expects {'all': {...}}
        print("DEBUG: data_full keys=", list(data_full.keys()))
        if 'all' not in data_full:
            raise RuntimeError("Dataset_JokeLoader.load() must return {'all': {...}}")
        all_data = data_full['all']  # {'raw': [...], 'tokens': [...], maybe 'raw'?}
        print(f"DEBUG: num_raw={len(all_data.get('raw', []))}, num_tokens={len(all_data.get('tokens', []))}")

        # 2) train the generator
        self.method.data = all_data  # pass tokens_list and raw
        train_res = self.method.run()  # returns {'train_loss': [...]}

        # 3) sample continuations
        samples = {}
        # five-word seeds (recommended by professor)
        five_word_seeds = [
            ('what', 'did', 'the', 'bartender', 'say'),
            ('why', 'did', 'the', 'chicken', 'cross'),
            ('i', 'asked', 'the', 'librarian', 'if'),
            ('when', 'life', 'gives', 'you', 'lemons'),
            ('once', 'upon', 'a', 'time', 'there')
        ]
        for seed in five_word_seeds:
            key = '5:' + ' '.join(seed)
            samples[key] = self.method.generate(list(seed), max_gen_len=50)

        # optional: also show a few 3-word seeds for contrast
        three_word_seeds = [('what', 'did', 'the'), ('why', 'did', 'the')]
        for seed in three_word_seeds:
            key = '3:' + ' '.join(seed)
            samples[key] = self.method.generate(list(seed), max_gen_len=50)

        # 4) save samples
        self.result.data = {**train_res, 'samples': samples}
        self.result.save()

        # 5) no numeric eval for generation
        return None, None
