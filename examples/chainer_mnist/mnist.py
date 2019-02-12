#!/usr/bin/env python
import chainer
import numpy as np
from teras import training
from teras.app import App, arg
from tqdm import tqdm


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(None, n_units)
            self.l2 = chainer.links.Linear(None, n_units)
            self.l3 = chainer.links.Linear(None, n_out)

    def forward(self, x):
        h1 = chainer.functions.relu(self.l1(x))
        h2 = chainer.functions.relu(self.l2(h1))
        return self.l3(h2)


def set_chainer_train(enable=True):
    chainer.config.train = enable
    chainer.config.enable_backprop = enable


chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)


def train(n_epoch=20, batch_size=100, n_units=1000, device=-1):
    train, test = chainer.datasets.get_mnist()
    train_x, train_y = [np.array(cols) for cols in zip(*train)]
    test_x, test_y = [np.array(cols) for cols in zip(*test)]

    model = MLP(n_units, 10)
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu(device)
    optimizer = chainer.optimizers.Adam().setup(model)

    trainer = training.Trainer(
        optimizer, model,
        loss_func=chainer.functions.softmax_cross_entropy,
        accuracy_func=chainer.functions.accuracy)
    trainer.configure(hooks={
        training.EPOCH_TRAIN_BEGIN: lambda _: set_chainer_train(True),
        training.EPOCH_VALIDATE_BEGIN: lambda _: set_chainer_train(False)
    }, converter=lambda x: chainer.dataset.convert.to_device(device, x))
    trainer.add_listener(
        training.listeners.ProgressBar(lambda n: tqdm(total=n)), priority=200)
    trainer.fit((train_x, train_y), (test_x, test_y), n_epoch, batch_size)


if __name__ == "__main__":
    App.configure(name='chainer-mnist', logoption='d')
    App.add_command('train', train, {
        'batch_size':
        arg('--batchsize', '-b', type=int, default=100,
            help='Number of images in each mini-batch'),
        'device':
        arg('--device', type=int, default=-1, metavar='ID',
            help='Device ID (negative value indicates CPU)'),
        'n_epoch':
        arg('--epoch', '-e', type=int, default=20,
            help='Number of sweeps over the dataset to train'),
        'n_units':
        arg('--unit', '-u', type=int, default=1000,
            help='Number of units'),
    }, description="Execute training")
    App.run()
