#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F

from teras.app import App, arg
from teras.framework.chainer import config as chainer_config
from teras.framework.chainer.model import MLP
import teras.logging as Log
from teras.training import Trainer


def train(n_epoch=20,
          batch_size=100,
          n_layers=3,
          n_units=1000,
          dropout=0.2,
          gpu=-1):

    train, test = chainer.datasets.get_mnist()
    train_x, train_y = train._datasets
    test_x, test_y = test._datasets

    Log.v('')
    Log.i('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(MLP))
    Log.i('# layer: {}'.format(n_layers))
    Log.i('# unit: {}'.format(n_units))
    Log.i('# dropout: {}'.format(dropout))
    Log.i('--------------------------------')
    Log.v('')

    assert n_layers >= 1
    layers = [MLP.Layer(None, n_units, F.relu, dropout)
              for i in range(n_layers - 1)]
    layers.append(MLP.Layer(None, 10))
    model = MLP(layers)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    Log.i('optimizer: Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)')

    trainer = Trainer(optimizer, model, loss_func=F.softmax_cross_entropy,
                      accuracy_func=F.accuracy)
    trainer.configure(chainer_config)
    trainer.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=(test_x, test_y))


def decode():
    pass


App.add_command('train', train, {
    'batch_size':
    arg('--batchsize', '-b', type=int, default=100,
        help='Number of examples in each mini-batch',
        metavar='SIZE'),
    'gpu':
    arg('--gpu', '-g', type=int, default=-1,
        help='use gpu device', metavar='DEVICE'),
    'dropout':
    arg('--dropout', '-dr', type=float, default=0.2,
        help='dropout ratio', metavar='RATIO'),
    'n_epoch':
    arg('--epoch', '-e', type=int, default=20,
        help='Number of sweeps over the dataset to train',
        metavar='NUM'),
    'n_layers':
    arg('--layer', '-l', type=int, default=3,
        help='Number of layers', metavar='NUM'),
    'n_units':
    arg('--unit', '-u', type=int, default=1000,
        help='Number of units', metavar='NUM')
}, description="exec train")

App.add_command('decode', decode, {})


if __name__ == "__main__":
    App.configure(name='chainer-mnist', loglevel=Log.DISABLE)
    App.run()
