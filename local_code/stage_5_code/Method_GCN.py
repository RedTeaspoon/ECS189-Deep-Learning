
from local_code.base_class.method import method
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class Method_GCN(method, nn.Module):
    def __init__(self, mName, mDescription, num_features, num_classes,
                 lr=1e-2, epochs=100, patience=10, num_layers=2, hidden_dim=16,
                 optimizer_name='adam', loss_name='CrossEntropy', label_smoothing=0.1, dropout_p=0.5  ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

      self.dropout_p = dropout_p
        # define a nn.Dropout module
        self.dropout = nn.Dropout(p=self.dropout_p)
        #—— Configuration ——#
        self.num_features   = num_features
        self.num_classes    = num_classes
        self.num_layers     = num_layers
        self.hidden_dim     = hidden_dim
        self.lr             = lr
        self.optimizer_name = optimizer_name
        self.loss_name      = loss_name
        self.label_smoothing = label_smoothing
        self.epochs         = epochs
        self.patience       = patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.num_layers not in (2, 3):
            raise ValueError("num_layers must be 2 or 3.")

        # First layer weight: [num_features × hidden_dim]
        self.gc1_weight = Parameter(torch.FloatTensor(num_features, hidden_dim))

	# Choose from 2 or 3 layers
        if num_layers == 2:
            # Second layer weight: [hidden_dim × num_classes]
            self.gc2_weight = Parameter(torch.FloatTensor(hidden_dim, num_classes))
            self.gc3_weight = None

        else:  # num_layers == 3
            # Second layer is hidden→hidden
            self.gc2_weight = Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # Third layer is hidden→num_classes
            self.gc3_weight = Parameter(torch.FloatTensor(hidden_dim, num_classes))
        self.reset_parameters()

        # Metrics buffers
        self.epochs_list     = []
        self.train_loss_list = []
        self.train_acc_list  = []
        self.val_loss_list = []
        self.val_acc_list   = []

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.gc1_weight.size(1))
        self.gc1_weight.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.gc2_weight.size(1))
        self.gc2_weight.data.uniform_(-stdv2, stdv2)

        # -----------------------------------------
        if self.num_layers == 3 and self.gc3_weight is not None:
            std3 = 1.0 / math.sqrt(self.gc3_weight.size(1))
            self.gc3_weight.data.uniform_(-std3, std3)

        # -----------------------------------------

    def forward(self, input, adj):
        # Layer 1
        support1 = torch.mm(input, self.gc1_weight)
        x = torch.spmm(adj, support1)
        x = F.relu(x)
        x = self.dropout(x)

        # --------------Choose from 2 or 3 layers-------------------------
        if self.num_layers == 2:
            # -- Layer 2 (output) --
            support2 = torch.mm(x, self.gc2_weight)  # [N×num_classes]
            output = torch.spmm(adj, support2)  # [N×num_classes]
            return output

        else:  # num_layers == 3
            # -- Layer 2 (hidden → hidden) --
            support2 = torch.mm(x, self.gc2_weight)  # [N×hidden_dim]
            x2 = torch.spmm(adj, support2)  # [N×hidden_dim]
            x2 = F.relu(x2)
            x2 = self.dropout(x2)

            # -- Layer 3 (hidden → classes) --
            support3 = torch.mm(x2, self.gc3_weight)  # [N×num_classes]
            output = torch.spmm(adj, support3)  # [N×num_classes]
            #---------------------------------------

        return output

    def run(self):
        graph, train_test_val = self.data['graph'], self.data['train_test_val']

        # unpack everything from graph & train_test_val
        adj = graph['utility']['A'].to(self.device)
        features = graph['X'].to(self.device)
        labels = graph['y'].to(self.device)
        adj = graph['utility']['A'].to(self.device)

        idx_train = train_test_val['idx_train'].to(self.device)
        idx_val = train_test_val['idx_val'].to(self.device)
        idx_test = train_test_val['idx_test'].to(self.device)

        # -------------------------------------
        # —— Select optimizer —— #
        opt_name = self.optimizer_name.lower()
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=5e-4)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer_name = {self.optimizer_name}. Must be Adam/SGD/RMSprop.")

        # —— Select loss function —— #
        loss_name = self.loss_name.lower()
        if loss_name == 'crossentropy':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_name == 'labelsmoothing':
            # PyTorch ≥1.10 supports label_smoothing in CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            raise ValueError(f"Unknown loss_name = {self.loss_name}. Must be CrossEntropy or LabelSmoothing.")
        # -------------------------------------

        #-----------set early stop variables----------
        best_loss = float('inf')
        best_state = None
        wait = 0

        # Training loop
        for ep in range(1, self.epochs + 1):
            # Train
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(features, adj)
            loss = loss_fn(y_pred[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            preds_train = y_pred[idx_train].argmax(dim=1)
            train_acc = (preds_train == labels[idx_train]).sum().item() / idx_train.shape[0]

            # Validate
            self.eval()
            with torch.no_grad():
                # y_pred = self.forward(features, adj)
                val_loss = loss_fn(y_pred[idx_val], labels[idx_val]).item()
                preds_val = y_pred[idx_val].argmax(dim=1)
                val_acc = (preds_val == labels[idx_val]).sum().item() / idx_val.shape[0]
            
            # ---- Print epoch summary------
            print(f"Epoch {ep}/{self.epochs} — "
                  f"Loss: {train_loss:.4f} — Acc: {train_acc:.4f} — "
                  f"Val Loss: {val_loss:.4f} — Test Acc: {val_acc:.4f}")
            
            # --------Per-epoch macro P/R/F1--------
            epoch_eval = Evaluate_Accuracy('epoch_eval', '')
            epoch_eval.data = {
                'true_y': labels[idx_train].cpu().numpy(),
                'pred_y': preds_train.cpu().numpy()
            }
            epoch_eval.evaluate()

            epoch_eval.data = {
                'true_y': labels[idx_val].cpu().numpy(),
                'pred_y': preds_val.cpu().numpy()
            }
            epoch_eval.evaluate()

            # Record
            self.epochs_list.append(ep)
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.val_acc_list.append(val_acc)
            self.val_loss_list.append(val_loss)

            # ------Early stopping--------
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {ep}/{self.epochs} (no improvement)", flush=True)
                    break
        
        # Restore best model
        if best_state is not None:
            self.load_state_dict(best_state)

        # Final test
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(features, adj)
            preds_test = y_pred[idx_test].argmax(dim=1)
            test_acc = (preds_test == labels[idx_test]).sum().item() / preds_test.shape[0]

        test_eval = Evaluate_Accuracy('test_eval', '')
        test_eval.data = {
            'true_y': labels[idx_test].cpu().numpy(),
            'pred_y': preds_test.cpu().numpy()
        }
        test_eval.evaluate()

        # Return results
        return {
            'true_y': labels[idx_test].cpu().numpy(),
            'pred_y': preds_test.cpu().numpy(),
            'epochs': self.epochs_list,
            'train_loss': self.train_loss_list,
            'train_acc': self.train_acc_list,
            'val_loss': self.val_loss_list,
            'val_acc': self.val_acc_list,
            'test_acc': test_acc
        }
