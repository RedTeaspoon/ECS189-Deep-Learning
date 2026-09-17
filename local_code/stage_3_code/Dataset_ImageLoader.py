'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle,os

class Dataset_ImageLoader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, file_name=None):
        super().__init__(dName, dDescription)
        self.dataset_source_file_name = file_name
        self.dataset_source_folder_path = None
    
    def load(self):
        print(f'loading data from {self.dataset_source_file_name}...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        X_train = [inst['image'] for inst in raw['train']]
        y_train = [inst['label'] for inst in raw['train']]
        X_test = [inst['image'] for inst in raw['test']]
        y_test = [inst['label'] for inst in raw['test']]
        #-------------set ORL for correct run on cuda---------
        if self.dataset_source_file_name.upper() == 'ORL':
            y_train = [l - 1 for l in y_train]
            y_test  = [l - 1 for l in y_test]
        #-----------------------------------------------------
        return {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}}

