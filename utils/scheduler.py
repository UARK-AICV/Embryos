from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LambdaLR
from torch.utils.data import DataLoader, random_split
import math 

class WarmupCosineScheduler(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to LR over `warmup_steps` training steps.
        Decreases learning rate from LR to 0 over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, verbose=True):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch, verbose=verbose)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
