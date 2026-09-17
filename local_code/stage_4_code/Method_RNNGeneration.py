
from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Method_RNNGeneration(method, nn.Module):
    """
    RNN-based text generator. Trains on token sequences for next-word prediction,
    and can generate new text given a seed sequence.
    """

    def __init__(self, mName, mDescription,
                 vocab,
                 embed_dim=100,
                 hidden_dim=512,
                 num_layers=2,
                 lr=1e-3,
                 batch_size=256,
                 epochs=1000,
                 max_seq_len=50):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Config
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_len = max_seq_len
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Model
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=vocab.get('<PAD>', 0))
        '''
        #------------RNN---------------
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        '''
        '''
        #------------LSTM-------------
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        '''
        #'''
        #------------GRU---------------
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        #'''



        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)
        self.to(self.device)

        # History
        self.loss_list = []
        self.patience = 5

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        emb = self.embedding(x.to(self.device))  # [B, L, E]
        out, hidden = self.rnn(emb, hidden)  # [B, L, H]
        logits = self.fc_out(out)  # [B, L, V]
        return logits, hidden

    def run(self):
        # Prepare data
        # self.data is expected to be a dict with 'raw' and 'tokens'
        data = self.data
        tokens_list = data['tokens']

        # Build dataset for next-word prediction
        class SeqDS(Dataset):
            def __init__(self, tokens_list, vocab, max_len):
                self.vocab = vocab
                self.max_len = max_len
                self.inputs = []
                self.targets = []
                for tokens in tokens_list:
                    idxs = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens]
                    # Truncate/pad
                    if len(idxs) < max_len + 1:
                        idxs = idxs + [vocab.get('<PAD>', 0)] * (max_len + 1 - len(idxs))
                    else:
                        idxs = idxs[:max_len + 1]
                    self.inputs.append(idxs[:-1])
                    self.targets.append(idxs[1:])

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return torch.tensor(self.inputs[idx], dtype=torch.long), \
                    torch.tensor(self.targets[idx], dtype=torch.long)

        ds = SeqDS(tokens_list, self.vocab, self.max_seq_len)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.get('<PAD>', 0))
        #--------------Adam----------------
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #--------------SGD-----------------
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        #-------------RMSprop--------------
        #optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.99)

        best_loss = float('inf')
        wait = 0
        # Training loop
        for ep in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0
            for Xb, yb in dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits, _ = self(Xb)
                # reshape for loss: [B*L, V] vs [B*L]
                B, L, V = logits.size()
                loss = criterion(logits.view(B * L, V), yb.view(B * L))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * B
            avg_loss = total_loss / len(ds)
            self.loss_list.append(avg_loss)
            print(f"Epoch {ep}/{self.epochs} — Train Loss: {avg_loss:.4f}")
            #--------early stopipng check------------
            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {ep}/{self.epochs} "
                          f"(no improvement in {self.patience} epochs)")
                    break

        return {'train_loss': self.loss_list}

    def generate(self, seed_words, max_gen_len=50, temperature=1.0):
        """
        Generate a sequence of words given a list of seed words.
        """
        self.eval()
        # convert seed to indices
        idxs = [self.vocab.get(w, self.vocab.get('<UNK>', 1)) for w in seed_words]
        idxs = idxs[-self.max_seq_len:]
        # initialize hidden state
        hidden = None
        # feed seed
        input_seq = torch.tensor([idxs], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits, hidden = self(input_seq)
        generated = []
        last_idx = idxs[-1]
        for _ in range(max_gen_len):
            inp = torch.tensor([[last_idx]], dtype=torch.long).to(self.device)
            logits, hidden = self(inp, hidden)
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
            last_idx = probs.argmax(1).item()  # greedy
            word = next((w for w, i in self.vocab.items() if i == last_idx), '<UNK>')
            generated.append(word)
            if word == '<PAD>':
                break
        return ' '.join(generated)

