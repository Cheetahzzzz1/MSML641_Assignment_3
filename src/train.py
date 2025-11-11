"""
train.py
- Run experiments grid and save metrics.csv + plots into results/
Usage:
    python src/train.py --output_dir ./results --subset 20000
"""
import os, time, argparse, random
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from preprocess import Preprocessor
from utils import IMDBDataset, macro_f1
from models import SentimentRNN
import pandas as pd
import matplotlib.pyplot as plt

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start = time.time()
    for x,y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()*x.size(0)
        all_preds.extend((preds.detach().cpu().numpy()>0.5).astype(int).tolist())
        all_labels.extend(y.detach().cpu().numpy().astype(int).tolist())
    epoch_time = time.time() - start
    acc = sum([a==b for a,b in zip(all_preds, all_labels)])/len(all_labels)
    f1 = macro_f1(np.array(all_labels), np.array(all_preds))
    return total_loss/len(loader.dataset), acc, f1, epoch_time

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            l = criterion(p, y)
            total_loss += l.item()*x.size(0)
            preds.extend((p.detach().cpu().numpy()>0.5).astype(int).tolist())
            labels.extend(y.detach().cpu().numpy().astype(int).tolist())
    acc = sum([a==b for a,b in zip(preds, labels)])/len(labels)
    f1 = macro_f1(np.array(labels), np.array(preds))
    return total_loss/len(loader.dataset), acc, f1

def run_experiments(output_dir, device='cpu', subset=None):
    os.makedirs(output_dir, exist_ok=True)
    prep = Preprocessor()
    prep.prepare()
    if subset is not None:
        prep.df = prep.df.iloc[:subset].reset_index(drop=True)
    train_df, test_df = prep.split50()

    seq_lengths = [25,50,100]
    architectures = ['RNN','LSTM','BiLSTM']
    activations = ['tanh']           # keep fixed for assignment baseline (can be extended)
    optimizers = ['Adam']            # baseline
    grad_clip_options = [None, 1.0]

    results = []
    for arch in architectures:
        for seq in seq_lengths:
            for grad_clip in grad_clip_options:
                # prepare dataset
                train_ds = IMDBDataset(train_df['tokens'], train_df['label'].values, prep.vocab, seq)
                test_ds  = IMDBDataset(test_df['tokens'], test_df['label'].values, prep.vocab, seq)
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

                is_bidir = (arch == 'BiLSTM')
                arch_name = 'LSTM' if arch == 'BiLSTM' else arch

                model = SentimentRNN(vocab_size=len(prep.vocab)+1, embed_dim=100, hidden_size=64, num_layers=2, dropout=0.4, arch=arch_name, bidirectional=is_bidir, activation='tanh')
                model.to(device)
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                epoch_times = []
                train_losses = []
                train_accs = []
                train_f1s = []
                EPOCHS = 5   # assignment may ask for more; small default for fast runs
                for epoch in range(EPOCHS):
                    tloss, tacc, tf1, etime = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
                    epoch_times.append(etime)
                    train_losses.append(tloss); train_accs.append(tacc); train_f1s.append(tf1)

                test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
                res = {
                    'Model': arch,
                    'Activation': 'tanh',
                    'Optimizer': 'Adam',
                    'Seq Length': seq,
                    'Grad Clipping': 'Yes' if grad_clip else 'No',
                    'Accuracy': round(test_acc,4),
                    'F1_macro': round(test_f1,4),
                    'Epoch Time (s)': round(sum(epoch_times)/len(epoch_times),3)
                }
                results.append(res)

                # save last loss history plot for this config
                try:
                    plot_path = os.path.join(output_dir,'plots', f"loss_{arch}_{seq}_{'clip' if grad_clip else 'noclip'}.png")
                    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                    plt.figure()
                    plt.plot(train_losses, marker='o')
                    plt.title(f"Train Loss - {arch} seq{seq} gradclip={'Yes' if grad_clip else 'No'}")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.grid(True)
                    plt.savefig(plot_path)
                    plt.close()
                except Exception:
                    pass

    # save metrics
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # accuracy vs seq plots by model
    for model_name, g in df.groupby("Model"):
        plt.figure()
        plt.plot(g["Seq Length"].astype(int), g["Accuracy"], marker='o')
        plt.title(f"Accuracy vs Seq Length - {model_name}")
        plt.xlabel("Sequence Length")
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir,'plots', f"acc_vs_seq_{model_name}.png"))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(__file__), '..', 'results'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--subset', type=int, default=None, help='Use subset of dataset for quick runs (e.g., 20000)')
    args = parser.parse_args()
    run_experiments(args.output_dir, device=args.device, subset=args.subset)
