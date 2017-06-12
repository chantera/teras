#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import chainer
import chainer.functions as F
from progressbar import ProgressBar

from teras.app import App, arg
from teras.app.trainer import TrainEvent as Event, Trainer
from teras.dataset import Dataset
from teras.framework.chainer.model import MLP
import teras.logging as Log


# class Resource(object):
#     pass
#
#
# class Trainer(teras.app.Trainer):
#
#     def _process_train(self, data):
#         size = len(train_dataset)
#         batch_count = 0
#         loss = 0.0
#         accuracy = 0.0
#         p = ProgressBar(min_value=0, max_value=size, fd=sys.stderr).start()
#         for batch_index, batch in enumerate(train_dataset.batch(batch_size, colwise=True, shuffle=True)):
#             p.update((batch_size * batch_index) + 1)
#             batch_loss = self.
#         p.finish()
#
#     def _process_test(self, data):
#         size = len(train_dataset)
#         batch_count = 0
#         loss = 0.0
#         accuracy = 0.0
#         p = ProgressBar(min_value=0, max_value=size, fd=sys.stderr).start()
#         for batch_index, batch in enumerate(train_dataset.batch(batch_size, colwise=True, shuffle=True)):
#             p.update((batch_size * batch_index) + 1)
#             batch_loss = self.
#         p.finish()

#
# Resource.load_model()
# Resource.load_data()


# train = Trainer(lazy_loader=Resource)
#


def train(n_epoch=20,
          batch_size=100,
          n_layers=3,
          n_units=1000,
          dropout=0.2,
          gpu=-1,
          debug=False):

    # load dataset
    train, test = chainer.datasets.get_mnist()
    train_x, train_y = train._datasets
    test_x, test_y = test._datasets
    train_dataset = Dataset(train_x, train_y)
    test_dataset = Dataset(test_x, test_y)

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(MLP))
    Log.i('# layer: {}'.format(n_layers))
    Log.i('# unit: {}'.format(n_units))
    Log.i('# dropout: {}'.format(dropout))
    Log.v('--------------------------------')
    Log.v('')

    # set up a neural network model
    assert n_layers >= 1
    layers = [MLP.Layer(None, n_units, F.relu, dropout)
              for i in range(n_layers - 1)]
    layers.append(MLP.Layer(None, 10))
    model = MLP(layers)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    chainer.config.use_cudnn = 'auto'
    if debug:
        chainer.config.debug = True
        chainer.config.type_check = True
    else:
        chainer.config.debug = False
        chainer.config.type_check = False

    # set up an optimizer
    optimizer = chainer.optimizers.Adam(
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    Log.i('optimizer: Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)')

    for epoch in range(n_epoch):
        # Training
        chainer.config.train = True
        chainer.config.enable_backprop = True

        size = len(train_dataset)
        batch_count = 0
        loss = 0.0
        accuracy = 0.0

        p = ProgressBar(min_value=0, max_value=size,
                        fd=sys.stderr).start()
        for i, (x, t) in enumerate(
                train_dataset.batch(batch_size, colwise=True, shuffle=True)):
            p.update((batch_size * i) + 1)
            batch_count += 1
            # forward
            y = model(x)
            batch_loss = F.softmax_cross_entropy(y, t)
            batch_accuracy = F.accuracy(y, t)
            loss += batch_loss.data
            accuracy += batch_accuracy.data
            # update
            optimizer.target.cleargrads()
            batch_loss.backward()
            optimizer.update()
            del batch_loss
        p.finish()
        Log.i("[training] epoch %d - #samples: %d, loss: %f, accuracy: %f" %
              (epoch + 1, size, loss / batch_count, accuracy / batch_count))

        # Evaluation
        chainer.config.train = False
        chainer.config.enable_backprop = False

        size = len(test_dataset)
        batch_count = 0
        loss = 0.0
        accuracy = 0.0

        p = ProgressBar(min_value=0, max_value=size,
                        fd=sys.stderr).start()
        for i, (x, t) in enumerate(
                test_dataset.batch(batch_size, colwise=True, shuffle=False)):
            p.update((batch_size * i) + 1)
            batch_count += 1
            # forward
            y = model(x)
            batch_loss = F.softmax_cross_entropy(y, t)
            batch_accuracy = F.accuracy(y, t)
            loss += batch_loss.data
            accuracy += batch_accuracy.data
        p.finish()
        Log.i("[evaluation] epoch %d - #samples: %d, loss: %f, accuracy: %f" %
              (epoch + 1, size, loss / batch_count, accuracy / batch_count))

        Log.v('-')


def train2(n_epoch=20,
           batch_size=100,
           n_layers=3,
           n_units=1000,
           dropout=0.2,
           gpu=-1,
           debug=False):

    # load dataset
    train, test = chainer.datasets.get_mnist()
    train_x, train_y = train._datasets
    test_x, test_y = test._datasets
    # train_dataset = Dataset(train_x, train_y)
    # test_dataset = Dataset(test_x, test_y)

    Log.v('')
    Log.v("initialize ...")
    Log.v('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# gpu: {}'.format(gpu))
    Log.i('# model: {}'.format(MLP))
    Log.i('# layer: {}'.format(n_layers))
    Log.i('# unit: {}'.format(n_units))
    Log.i('# dropout: {}'.format(dropout))
    Log.v('--------------------------------')
    Log.v('')

    assert n_layers >= 1
    layers = [MLP.Layer(None, n_units, F.relu, dropout)
              for i in range(n_layers - 1)]
    layers.append(MLP.Layer(None, 10))
    model = MLP(layers)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    chainer.config.use_cudnn = 'auto'
    if debug:
        chainer.config.debug = True
        chainer.config.type_check = True
    else:
        chainer.config.debug = False
        chainer.config.type_check = False

    def chainer_train_on(*args, **kwargs):
        chainer.config.train = True
        chainer.config.enable_backprop = True

    def chainer_train_off(*args, **kwargs):
        chainer.config.train = False
        chainer.config.enable_backprop = False

    optimizer = chainer.optimizers.Adam(
        alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    Log.i('optimizer: Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)')

    trainer = Trainer(optimizer, model, loss_func=F.softmax_cross_entropy)
    trainer.add_hook(Event.EPOCH_TRAIN_BEGIN, chainer_train_on)
    trainer.add_hook(Event.EPOCH_VALIDATION_BEGIN, chainer_train_off)
    trainer.add_hook(Event.EPOCH_END, lambda x: Log.v('-'))
    trainer.fit(train_x, train_y,
                batch_size=batch_size,
                validation_data=(test_x, test_y))


def decode():
    pass


App.add_command('train', train2, {
    'batch_size':
    arg('--batchsize', '-b', type=int, default=100,
        help='Number of examples in each mini-batch'),
    'gpu':
    arg('--gpu', '-g', type=int, default=-1),
    'dropout':
    arg('--dropout', '-dr', type=float, default=0.2,
        help='dropout ratio'),
    'n_epoch':
    arg('--epoch', '-e', type=int, default=20,
        help='Number of sweeps over the dataset to train'),
    'n_layers':
    arg('--layer', '-l', type=int, default=3, help='Number of layers'),
    'n_units':
    arg('--unit', '-u', type=int, default=1000, help='Number of units')
}, description="execute train")

App.add_command('decode', decode, {})

App.add_arg('debug', True)
App.configure(loglevel=Log.DISABLE)


if __name__ == "__main__":
    App.run()
