import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os 
import os.path as osp

__all__ = ['ConfusionMatrix']


class ConfusionMatrix():
    def __init__(self, nclasses, print=True, savefig_dir=None):
        self.nclasses = nclasses
        self.print = print
        self.savefig_dir = savefig_dir
        self.reset()

    def update(self, output, target, is_prob=True):
        if is_prob:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
        self.cm += confusion_matrix(target.cpu().numpy(),
                                    pred.cpu().numpy(),
                                    labels=range(self.nclasses))

    def reset(self):
        self.cm = np.zeros(shape=(self.nclasses, self.nclasses))

    def value(self):
        return self.cm

    def report(self):
        return str(self.cm)

    def summary(self, tag="-", logger=None):
        func = logger.info if logger is not None else print
        func('+ Confusion matrix: ')
        if self.print:
            func(self.cm)
        if self.savefig_dir is not None:
            os.makedirs(self.savefig_dir, exist_ok=True)
            df_cm = pd.DataFrame(self.cm,
                                 index=range(self.nclasses),
                                 columns=range(self.nclasses))
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True, cmap='YlGnBu')
            plt.tight_layout()
            plt.savefig(self.savefig_dir + f'/cm-{tag}.png')
