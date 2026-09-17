'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import csv,os, re

class Dataset_JokeLoader(dataset):
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
        """
        Load all jokes as strings and token lists.
        Returns:
            {'all': {'raw': List[str], 'tokens': List[List[str]]}}
        """
        path = os.path.join(self.dataset_source_folder_path, self.dataset_root)
        raw_texts, tokens_list = [], []

        if os.path.isdir(path):
            # Case A: many .txt files
            for fname in sorted(os.listdir(path)):
                if fname.lower().endswith('.txt'):
                    with open(os.path.join(path, fname), encoding='utf8') as f:
                        txt = f.read().strip()
                    raw_texts.append(txt)
                    tokens_list.append(self._clean_and_tokenize(txt))
        elif os.path.isfile(path):
            # Case B: single CSV of jokes
            with open(path, encoding='utf8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    txt = row.get('Joke', '')
                    raw_texts.append(txt)
                    tokens_list.append(self._clean_and_tokenize(txt))
        else:
            raise FileNotFoundError(f"No file or folder found at {path}")

        if not raw_texts:
            raise RuntimeError(f"No jokes loaded from {path}")

        return {'all': {'raw': raw_texts, 'tokens': tokens_list}}
