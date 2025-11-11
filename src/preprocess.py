"""
preprocess.py
- Loads `data/IMDB Dataset.csv`
- Applies preprocessing rules from the assignment PDF:
  lowercase, remove punctuation, remove HTML tags, whitespace tokenize, keep top 10k vocab
- Exposes Preprocessor class to build vocab and split data (50/50)
"""
import re
import json
from collections import Counter
import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'IMDB Dataset.csv')

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_vocab(token_lists, max_vocab=10000):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    most_common = [w for w,_ in counter.most_common(max_vocab)]
    vocab = {w:i+1 for i,w in enumerate(most_common)}  # 0 reserved for pad/unk
    return vocab

class Preprocessor:
    def __init__(self, csv_path=None, max_vocab=10000):
        csv_path = csv_path or RAW_PATH
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.max_vocab = max_vocab
        self.vocab = None

    def prepare(self):
        self.df['clean'] = self.df['review'].apply(clean_text)
        self.df['label'] = (self.df['sentiment'] == 'positive').astype(int)
        self.df['tokens'] = self.df['clean'].str.split()
        self.vocab = build_vocab(self.df['tokens'], self.max_vocab)
        return self

    def split50(self):
        split = len(self.df)//2
        train = self.df.iloc[:split].reset_index(drop=True)
        test  = self.df.iloc[split:].reset_index(drop=True)
        return train, test

    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
