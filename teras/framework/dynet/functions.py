import numpy as np

import dynet

from teras.framework.dynet import _dynet_train


def dropout(x, ratio=0.5):
    if _dynet_train:
        mask = dynet.random_bernoulli(x, 1 - ratio)
        h_dropped = dynet.cmult(x, mask)
    else:
        # at test time, multiply by the retention rate to scale
        h_dropped = x * (1 - ratio)
    return h_dropped


def nll_loss(y, t):
    loss = dynet.pickneglogsoftmax_batch(y, t)
    loss = dynet.sum_batches(loss)
    loss.value()  # exec forward computation explicitly
    return loss


def accuracy(y, t):
    argmax = y.npvalue().argmax(axis=0)
    correct = np.sum(argmax == t)
    return correct / len(t)
