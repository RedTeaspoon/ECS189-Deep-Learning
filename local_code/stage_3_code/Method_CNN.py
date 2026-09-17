'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):

    """
    CNN: 2 conv layers + pooling, followed by 2 FC layers.
    Uses mini-batch training via DataLoader.
    """
    def __init__(self, mName, mDescription, in_channels=1, num_classes=10, img_size=28):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        #-------------------------
        # 1) unpack image size
        if isinstance(img_size, (tuple, list)):
            H, W = img_size
        else:
            H = W = img_size

        # 2) model definition
        #   – one conv→pool block
        # 3 layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        #   – compute flattened feature dimension

        flat_size = 32 * (H//2) * (W//2)

        #   – final linear
        self.fc1 = nn.Linear(flat_size, num_classes)

        #--------for plot----------
        # 4) placeholders & metric buffers
        self.train_loader   = None
        self.full_X_tensor  = None
        self.full_y_tensor  = None
        self._X_test        = None
        self._y_test        = None
        self.epochs = []
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        #----------------------
        # model
        """
        #-----------
        #5 layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        fc_input = 64 * (img_size // 2) * (img_size // 2)
        self.fc1   = nn.Linear(fc_input, 128)
        self.fc2   = nn.Linear(128, num_classes)
        #----------
        
        # 3 layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        flat_size = 16 * (img_size // 2) * (img_size // 2)
        self.fc1 = nn.Linear(flat_size, num_classes)
        """
        #---------
        # hyperparameters
        self.batch_size    = 32
        self.max_epoch     = 10
        self.learning_rate = 0.001
        self.patience = max(1, int(self.max_epoch * 0.1))
        #-----------Choosing device-----------------------
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print("Method_CNN using device:", self.device)
        self.to(self.device)
        #------------------------------------------------


    def forward(self, x):
        #--------------
        # 5 layers
        """
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        """
        # --------------
        # 3 layers
        #x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc1(x)

    def fit(self, X, y):
        # set train mode
        torch.nn.Module.train(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()
        evaluator = Evaluate_Accuracy('train_eval', '')
        # prepare data loader
        #arrX = np.array(X)
        if self.train_loader is None:
            # stack and convert only once
            arrX = np.stack(X, axis=0)
            if arrX.ndim == 4:  # got (N,H,W,3)
                if self.conv1.in_channels == 1:
                    # ORL case: just take the first channel
                    arrX = arrX[..., 0]  # now (N,H,W)
                else:
                    # CIFAR case: channel-last → channel-first
                    arrX = arrX.transpose(0, 3, 1, 2)  # (N,3,H,W)
            # add channel dimension
            if arrX.ndim == 3:
                X_tensor = torch.from_numpy(arrX).unsqueeze(1).float()
            else:
                X_tensor = torch.from_numpy(arrX).float()
            y_tensor = torch.LongTensor(y)
            # store full dataset tensors for evaluation
            self.full_X_tensor = X_tensor
            self.full_y_tensor = y_tensor

            #----------------
            arrX_test = np.stack(self.data['test']['X'], axis=0)  # shape (N_test, H, W)
            if arrX_test.ndim == 4 and self.conv1.in_channels == 1:
                arrX_test = arrX_test[..., 0]
            if arrX_test.ndim == 4:
                arrX_test = arrX_test.transpose(0, 3, 1, 2)

            if arrX_test.ndim == 3:
                X_test_tens = torch.from_numpy(arrX_test).unsqueeze(1).float()
            else:
                X_test_tens = torch.from_numpy(arrX_test).float()
            y_test_tens = torch.LongTensor(self.data['test']['y'])
            # move test to same device
            dev = next(self.parameters()).device
            X_test_tens = X_test_tens.to(dev)
            y_test_tens = y_test_tens.to(dev)
            self._X_test, self._y_test = X_test_tens, y_test_tens

            #-------------
            dataset = TensorDataset(X_tensor, y_tensor)
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )

        # early stopping parameters
        best_loss = float('inf')
        #patience = 3
        wait = 0

        # training loop
        for epoch in range(1, self.max_epoch + 1):
            running_loss = 0.0
            #-----bX: batch of images------
            #-----bY: batch of labels------
            for bX, bY in self.train_loader:
                bX, bY = bX.to(self.device), bY.to(self.device)
                optimizer.zero_grad()
                outputs = self(bX)
                loss = loss_func(outputs, bY)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)

            # evaluate on full train set
            with torch.no_grad():
                preds = self(self.full_X_tensor.to(self.device)).max(1)[1].cpu()
            evaluator.data = {
                'true_y': self.full_y_tensor,
                'pred_y': preds
            }

            acc_train = evaluator.evaluate()

            with torch.no_grad():
                logits = self(X_test_tens)
                loss_test = nn.CrossEntropyLoss()(logits, y_test_tens).item()
                preds_test = logits.max(1)[1].cpu()
            evaluator = Evaluate_Accuracy('test_eval', '')
            evaluator.data = {
                'true_y': y_test_tens.cpu().numpy(),  # or torch tensor
                'pred_y': preds_test.numpy()
            }
            test_acc = evaluator.evaluate()

            # record train metrics
            self.epochs.append(epoch)
            self.train_loss.append(avg_loss)
            self.train_acc.append(acc_train)
            self.test_loss.append(loss_test)
            self.test_acc.append(test_acc)
            # ---------------------------
            print(
                f'Epoch {epoch}/{self.max_epoch} — '
                f'Loss: {avg_loss:.4f} — Acc: {acc_train:.4f} — '
                f'Test Loss: {loss_test:.4f} — Test Acc: {test_acc:.4f}',
                flush=True
            )

            # early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch}/{self.max_epoch} (loss did not improve)",
                        flush=True
                    )
                    break

    def test(self, X):
        self.eval()
        arrX = np.stack(X, axis=0)

        if arrX.ndim == 4 and self.conv1.in_channels == 1:
            arrX = arrX[..., 0]
        if arrX.ndim == 4:
            arrX = arrX.transpose(0, 3, 1, 2)

        if arrX.ndim == 3:
            X_tensor = torch.from_numpy(arrX).unsqueeze(1).float()
        else:
            X_tensor = torch.from_numpy(arrX).float()
        X_tensor = X_tensor.to(self.device)

        with torch.no_grad():
            outputs = self(X_tensor)
            preds = outputs.max(1)[1].cpu()
        return preds

    def run(self):
        print('Method_CNN running...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        pred_y = self.test(self.data['test']['X'])

        return {
            'true_y': self.data['test']['y'],
            'pred_y': pred_y,
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'train_acc': self.train_acc,
            'test_acc': self.test_acc
        }
