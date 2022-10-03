import torch


__all__ = ['Accuracy']

class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def update(self, output, target, is_prob=True):
        if is_prob:
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            
        correct = (pred == target).sum()
        sample_size = output.size(0)
        self.correct += correct
        self.sample_size += sample_size

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def report(self):
        return f'{self.value():.4f}'

    def summary(self, logger=None):
        if logger is None:
            print(f'+ Accuracy: {self.value()}')
        else:
            logger.info(f'+ Accuracy: {self.value()}')
