'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name_train, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_train.append(elements[1:])
            y_train.append(elements[0])
        f.close()

        X_test= []
        y_test = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name_test, 'r')

        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X_test.append(elements[1:])
            y_test.append(elements[0])
        f.close()
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}