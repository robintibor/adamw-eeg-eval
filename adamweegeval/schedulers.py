import numpy as np

class ScheduledOptimizer(object):
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.initial_lrs = list(map(
            lambda group: group['lr'], optimizer.param_groups))
        self.initial_weight_decays = list(map(
            lambda group: group['weight_decay'], optimizer.param_groups))
        self.i_update = 0

    def step(self):
        for group, inital_lr, initial_wd in zip(
                self.optimizer.param_groups,
                self.initial_lrs,
                self.initial_weight_decays):
            group['lr'] = self.scheduler.get_lr(inital_lr, self.i_update)
            group['weight_decay'] = self.scheduler.get_weight_decay(
                initial_wd, self.i_update)
        self.optimizer.step()
        self.i_update += 1

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()


class CosineAnnealing(object):
    def __init__(self, n_updates_per_period, restart_after_end):
        self.n_updates_per_period = n_updates_per_period
        self.restart_after_end = restart_after_end

    def get_lr(self, initial_val, i_update):
        if i_update >= self.n_updates_per_period and self.restart_after_end == False:
            raise ValueError("More updates ({:d}) than expected ({:d})".format(
                i_update, self.n_updates_per_period))
        i_update = i_update % self.n_updates_per_period
        fraction_period = i_update / float(self.n_updates_per_period)
        return initial_val * (0.5 * np.cos(np.pi * fraction_period) + 0.5)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)
