#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from teras.app import App, arg
import teras.dataset
from teras.framework.pytorch import config as pytorch_config
import teras.logging as Log
from teras.training import Trainer
from teras.training.event import TrainEvent as Event


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(n_epoch=20,
          batch_size=100,
          lr=0.01,
          momentum=0.5,
          cuda=False,
          seed=1):
    torch.manual_seed(seed)

    train, test = teras.dataset.get_mnist()
    train_x = train[0].reshape((len(train[0]), 1, 28, 28))
    train_y = train[1].astype('int64')
    test_x = test[0].reshape((len(test[0]), 1, 28, 28))
    test_y = test[1].astype('int64')

    Log.v('')
    Log.i('--------------------------------')
    Log.i('# Minibatch-size: {}'.format(batch_size))
    Log.i('# epoch: {}'.format(n_epoch))
    Log.i('# use_cuda: {}'.format(cuda))
    Log.i('# model: {}'.format(Net))
    Log.i('--------------------------------')
    Log.v('')

    model = Net()
    if cuda and torch.cuda.is_available():
        model.cuda()
        pytorch_config['converter'] = \
            lambda x: torch.autograd.Variable(torch.from_numpy(x).cuda())
    else:
        pytorch_config['converter'] = \
            lambda x: torch.autograd.Variable(torch.from_numpy(x))

    def accuracy(y, t):
        pred = y.data.max(1)[1]
        correct = pred.eq(t.data).cpu().sum()
        return correct / len(t)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    Log.i('optimizer: SGD(lr={}, momentum={})'.format(lr, momentum))

    trainer = Trainer(optimizer, model, loss_func=F.nll_loss,
                      accuracy_func=accuracy)
    trainer.configure(pytorch_config)
    trainer.add_hook(Event.EPOCH_TRAIN_BEGIN, lambda data: model.train())
    trainer.add_hook(Event.EPOCH_VALIDATE_BEGIN, lambda data: model.eval())
    trainer.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=(test_x, test_y))


def decode():
    """define decode command"""
    pass


App.add_command('train', train, {
    'batch_size': arg('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size (default: 64)'),
    'n_epoch': arg('--epochs', type=int, default=10, metavar='N',
                   help='number of epochs to train (default: 10)'),
    'lr': arg('--lr', type=float, default=0.01, metavar='LR',
              help='learning rate (default: 0.01)'),
    'momentum': arg('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)'),
    'cuda': arg('--cuda', action='store_true', default=False,
                help='enables CUDA training'),
    'seed': arg('--seed', type=int, default=1, metavar='S',
                help='random seed (default: 1)')
}, description="PyTorch MNIST Example")

App.add_command('decode', decode, {})


if __name__ == "__main__":
    App.configure(name='pytorch-mnist', logoption='d')
    App.run()
