#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dynet

from teras.app import App, arg
import teras.dataset
from teras.framework.dynet import config as dynet_config
import teras.framework.dynet.functions as F
from teras.framework.dynet.model import MLP
import teras.logging as Log
from teras.training import Trainer
from teras.training.event import TrainEvent as Event


def train(n_epoch=20,
          batch_size=100,
          n_layers=3,
          n_units=1000,
          dropout=0.2):

    train, test = teras.dataset.get_mnist()
    train_x, train_y = train
    test_x, test_y = test

    Log.v('')
    Log.i('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# model: {}'.format(MLP))
    Log.i('# layer: {}'.format(n_layers))
    Log.i('# unit: {}'.format(n_units))
    Log.i('# dropout: {}'.format(dropout))
    Log.i('--------------------------------')
    Log.v('')

    assert n_layers >= 1
    if n_layers == 1:
        layers = [MLP.Layer(784, 10)]
    else:
        layers = [MLP.Layer(784, n_units, dynet.rectify, dropout)]
        for i in range(n_layers - 2):
            layers.append(MLP.Layer(n_units, n_units, dynet.rectify, dropout))
        layers.append(MLP.Layer(n_units, 10))
    model = MLP(layers)
    params = dynet.ParameterCollection()
    model.init_params(params)

    optimizer = dynet.AdamTrainer(
        params, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08, edecay=0.0)
    Log.i('optimizer: Adam(alpha=0.001, beta_1=0.9, beta_2=0.999, '
          'eps=1e-08, edecay=0.0)')

    def forward(x):
        x = dynet.inputTensor(x.T, batched=True)
        return model(x)

    trainer = Trainer(optimizer, forward, loss_func=F.nll_loss,
                      accuracy_func=F.accuracy)
    trainer.configure(dynet_config)
    trainer.add_hook(Event.EPOCH_TRAIN_END,
                     lambda data: optimizer.update_epoch())
    trainer.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=(test_x, test_y))


def decode():
    """define decode command"""
    pass


App.add_command('train', train, {
    'batch_size':
    arg('--batchsize', '-b', type=int, default=100,
        help='Number of examples in each mini-batch',
        metavar='SIZE'),
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
    App.configure(name='dynet-mnist', logoption='d')
    App.run()
