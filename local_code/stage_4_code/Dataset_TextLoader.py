'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle,os, re

class Dataset_TextLoader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_root = None

    def __init__(self, dName=None, dDescription=None, dataset_root=None):
        super().__init__(dName, dDescription)
        self.dataset_root = dataset_root
        self.dataset_source_folder_path = None

    def _clean_and_tokenize(self, text: str):
        # lowercase, remove non-letters, collapse spaces, split
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    def load(self):
        print(f'loading data from {self.dataset_root}...')
        data_splits = {}
        # determine root folder
        if self.dataset_root:
            root = os.path.join(self.dataset_source_folder_path, self.dataset_root)
        else:
            root = self.dataset_source_folder_path
        for split in ['train', 'test']:
            texts, labels = [], []
            split_dir = os.path.join(root, split)
            for sentiment in ['pos', 'neg']:
                lbl = 1 if sentiment == 'pos' else 0
                folder = os.path.join(split_dir, sentiment)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(f"Expected folder not found: {folder}")
                for fname in os.listdir(folder):
                    file_path = os.path.join(folder, fname)
                    with open(file_path, encoding='utf8') as f:
                        raw = f.read()
                    tokens = self._clean_and_tokenize(raw)
                    texts.append(tokens)
                    labels.append(lbl)
            data_splits[split] = {'X': texts, 'y': labels}
        
        return data_splits

