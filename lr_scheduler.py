import numpy as np

class LRScheduler:
    def __init__(self, initial_lr, strategy='exponential', **kwargs):
        self.lr = initial_lr
        self.strategy = strategy
        self.kwargs = kwargs

    def exponential_decay(self, epoch):
        """ 指数衰减 """
        decay_rate = self.kwargs.get('decay_rate', 0.95)
        return self.lr * (decay_rate ** epoch)

    def step_decay(self, epoch):
        """ 固定步长衰减 """
        drop = self.kwargs.get('drop', 0.5)
        epochs_drop = self.kwargs.get('epochs_drop', 10)
        return self.lr * (drop ** (epoch // epochs_drop))

    def multistep_decay(self, epoch):
        """ 多步长衰减 """
        milestones = self.kwargs.get('milestones', [20, 40, 60])
        drop = self.kwargs.get('drop', 0.5)
        for milestone in milestones:
            if epoch >= milestone:
                self.lr *= drop
        return self.lr

    def cosine_annealing(self, epoch, num_epochs):
        """ 余弦退火衰减 """
        return self.lr * (1 + np.cos(np.pi * epoch / num_epochs)) / 2

    def get_lr(self, epoch, num_epochs=None):
        if self.strategy == 'exponential':
            return self.exponential_decay(epoch)
        elif self.strategy == 'step':
            return self.step_decay(epoch)
        elif self.strategy == 'multistep':
            return self.multistep_decay(epoch)
        elif self.strategy == 'cosine':
            return self.cosine_annealing(epoch, num_epochs)
        else:
            raise ValueError("Unsupported learning rate decay strategy.")
