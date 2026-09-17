'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        acc = accuracy_score(y_true, y_pred)

        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        prec_micro = precision_score(y_true, y_pred, average='micro')
        rec_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_w = recall_score(y_true, y_pred, average='weighted')
        f1_w = f1_score(y_true, y_pred, average='weighted')

        # return with pipeline

        return {
            'accuracy': acc, 'precision_macro': prec_macro, 'recall_macro': rec_macro,
            'f1_macro': f1_macro, 'precision_micro': prec_micro, 'recall_micro': rec_micro,
            'f1_micro': f1_micro, 'precision_weighted': prec_w, 'recall_weighted': rec_w,
            'f1_weighted': f1_w
        }

        