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
        # unpack
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']

        # compute
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        rec_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        prec_weight = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_weight = recall_score(y_true, y_pred, average='weighted')
        f1_weight = f1_score(y_true, y_pred, average='weighted')

        print('evaluating performance......')
        print(f"Accuracy        : {acc:.4f}")
        print(f"Macro   P/R/F1  : {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
        print(f"Micro   P/R/F1  : {prec_micro:.4f} / {rec_micro:.4f} / {f1_micro:.4f}")
        print(f"Weighted P/R/F1 : {prec_weight:.4f} / {rec_weight:.4f} / {f1_weight:.4f}")

        # return only accuracy so Setting_Train_Test_Split still works
        return acc
