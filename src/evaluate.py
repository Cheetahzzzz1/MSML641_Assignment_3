"""
evaluate.py
- Create final plots from produced metrics.csv
Usage:
    python src/evaluate.py --metrics ./results/metrics.csv --out ./results/plots
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_accuracy_vs_seq(metrics_csv, out_dir):
    df = pd.read_csv(metrics_csv)
    os.makedirs(out_dir, exist_ok=True)
    for model, g in df.groupby('Model'):
        pivot = g.groupby('Seq Length')['Accuracy'].mean()
        plt.figure()
        pivot.plot(marker='o')
        plt.title(f'Accuracy vs Seq Length - {model}')
        plt.xlabel('Seq Length')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f'acc_vs_seq_{model}.png'))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.metrics), 'plots')
    plot_accuracy_vs_seq(args.metrics, out)
