"""
utils.py
- Dataset class and a small macro-F1 utility
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class IMDBDataset(Dataset):
    def __init__(self, token_lists, labels, vocab, seq_len):
        self.seqs = [self._to_seq(t, vocab, seq_len) for t in token_lists]
        self.labels = labels.astype(np.float32)

    def _to_seq(self, tokens, vocab, seq_len):
        ids = [vocab.get(w,0) for w in tokens][:seq_len]
        if len(ids) < seq_len:
            ids += [0]*(seq_len - len(ids))
        return np.array(ids, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

def macro_f1(true, pred):
    # true, pred: numpy arrays of 0/1
    f1s = []
    for cls in [0,1]:
        tp = int(((pred==cls) & (true==cls)).sum())
        fp = int(((pred==cls) & (true!=cls)).sum())
        fn = int(((pred!=cls) & (true==cls)).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    return float(sum(f1s)/len(f1s))
