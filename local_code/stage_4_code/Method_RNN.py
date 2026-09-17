
from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Method_RNN(method, nn.Module):
    """
    RNN-based classifier for tokenized text.
    """
    def __init__(self, mName, mDescription,
                 vocab,
                 embed_dim=100,
                 hidden_dim=128,
                 num_layers=1,
                 num_classes=2,
                 max_seq_len=200,
                 lr=1e-3,
                 batch_size=64,
                 epochs=5,
                 patience=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Config
        self.vocab       = vocab
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.test_loss_list = []
        self.patience = patience

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=vocab['<PAD>'])
        self.rnn       = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)
        self.to(self.device)

        # Metrics buffers
        self.epochs_list     = []
        self.train_loss_list = []
        self.train_acc_list  = []
        self.test_acc_list   = []

    def forward(self, x):
        # x: [batch, seq_len]
        emb, _ = self.rnn(self.embedding(x.to(self.device)))
        return self.fc(emb[:, -1, :])

    def run(self):
        # 1) Prepare DataLoaders
        data = self.data
        train_X, train_y = data['train']['X'], data['train']['y']
        test_X,  test_y  = data['test']['X'],  data['test']['y']

        class TextDS(Dataset):
            def __init__(self, X, y, vocab, max_len):
                self.X, self.y, self.vocab, self.max_len = X, y, vocab, max_len
            def __len__(self): return len(self.X)
            def __getitem__(self, idx):
                tokens = self.X[idx]
                idxs = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
                if len(idxs) < self.max_len:
                    idxs += [self.vocab['<PAD>']] * (self.max_len - len(idxs))
                else:
                    idxs = idxs[:self.max_len]
                return torch.tensor(idxs, dtype=torch.long), self.y[idx]

        train_ds = TextDS(train_X, train_y, self.vocab, self.max_seq_len)
        test_ds  = TextDS(test_X,  test_y,  self.vocab, self.max_seq_len)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_dl  = DataLoader(test_ds,  batch_size=self.batch_size)

        # 2) Optimizer & Loss
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        loss_fn   = nn.CrossEntropyLoss()

        #-----------set early stop variables----------
        best_loss = float('inf')
        wait = 0

        # 3) Training loop
        for ep in range(1, self.epochs + 1):
            # Train
            self.train()
            total_loss, total_correct, total = 0, 0, 0
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self(X_batch)
                loss   = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss   += loss.item() * y_batch.size(0)
                preds         = logits.argmax(1)
                total_correct += (preds == y_batch).sum().item()
                total        += y_batch.size(0)
            train_loss = total_loss / total
            train_acc  = total_correct / total

            # Evaluate
            self.eval()
            correct, total = 0, 0
            test_loss_sum = 0
            all_test_preds, all_test_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_dl:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = self(X_batch)
                    loss = loss_fn(logits, y_batch)
                    test_loss_sum += loss.item() * y_batch.size(0)
                    preds = logits.argmax(1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

                    # --------store for P/R/F1---------
                    all_test_preds.extend(preds.cpu().numpy().tolist())
                    all_test_true.extend(y_batch.cpu().numpy().tolist())

            test_loss = test_loss_sum / total
            test_acc = correct / total

            # ---- Print epoch summary------
            print(f"Epoch {ep}/{self.epochs} — "
                  f"Loss: {train_loss:.4f} — Acc: {train_acc:.4f} — "
                  f"Test Loss: {test_loss:.4f} — Test Acc: {test_acc:.4f}")

            # --------Per-epoch macro P/R/F1--------
            # print("evaluating performance......")
            epoch_eval = Evaluate_Accuracy('epoch_eval', '')
            epoch_eval.data = {
                'true_y': all_test_true,
                'pred_y': all_test_preds
            }
            epoch_eval.evaluate()

            # Record
            self.epochs_list.append(ep)
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            self.test_loss_list.append(test_loss)

            # ------Early stopping--------
            if test_loss <= best_loss:
                best_loss = test_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {ep}/{self.epochs} (no improvement)", flush=True)
                    break

        # 4) Final predictions
        all_true, all_pred = [], []
        self.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_dl:
                X_batch = X_batch.to(self.device)
                preds   = self(X_batch).argmax(1).cpu().numpy().tolist()
                all_pred.extend(preds)
                all_true.extend(y_batch.numpy().tolist())

        # 5) Return results
        return {
            'true_y':   np.array(all_true),
            'pred_y':   np.array(all_pred),
            'epochs':   self.epochs_list,
            'train_loss': self.train_loss_list,
            'train_acc':  self.train_acc_list,
            'test_loss': self.test_loss_list,
            'test_acc':   self.test_acc_list
        }
