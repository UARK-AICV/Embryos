import torch
from sklearn.metrics import roc_auc_score

__all__ = ['AUC']


class AUC():
    def __init__(self, nclasses, mode='macro', multi_class='ovo'):
        self.mode = mode
        self.multi_class = multi_class
        self.labels = list(range(nclasses))
        self.reset()

    def update(self, output, target):
        pred = torch.softmax(output, dim=1)
        self.pred += pred.cpu().tolist()
        self.target += target.cpu().tolist()

    def reset(self):
        self.pred = []
        self.target = []

    def value(self):
        return roc_auc_score(self.target, self.pred, labels=self.labels,
                             average=self.mode, multi_class=self.multi_class)

    def report(self):
        report_str = ''
        for mode in ['macro', 'weighted']:
            f1 = roc_auc_score(self.target, self.pred, labels=self.labels,
                               average=mode, multi_class=self.multi_class)
            report_str += f'{mode}: {f1}'
        
        return report_str

    def summary(self):
        print(f'+ AUC:')
        for mode in ['macro', 'weighted']:
            f1 = roc_auc_score(self.target, self.pred, labels=self.labels,
                               average=mode, multi_class=self.multi_class)
            print(f'{mode}: {f1}')


if __name__ == '__main__':
    auc = AUC(nclasses=4)
    auc.update(
        auc.calculate(
            torch.tensor([[.1, .2, .4, .3],
                          [.1, .1, .8, .0],
                          [.1, .5, .2, .2]]),
            torch.tensor([2, 2, 3])
        )
    )
    auc.summary()
    auc.update(
        auc.calculate(
            torch.tensor([[.9, .1, .0, .0],
                          [.6, .2, .1, .1],
                          [.7, .0, .3, .0]]),
            torch.tensor([1, 1, 2])
        )
    )
    auc.summary()
