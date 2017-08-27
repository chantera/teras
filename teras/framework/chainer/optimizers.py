class ExponentialDecayAnnealing(object):
    name = 'ExponentialDecayAnnealing'
    call_for_each_param = False

    def __init__(self, decay_rate, decay_step, staircase=False, lr_key='lr'):
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.staircase = staircase
        self.lr_key = lr_key
        self.step = 0

    def __call__(self, optimizer):
        self.step += 1
        lr = getattr(optimizer, self.lr_key)
        p = self.step / self.decay_step
        if self.staircase:
            p //= 1  # floor
        lr *= self.decay_rate ** p
        setattr(optimizer, self.lr_key, lr)
