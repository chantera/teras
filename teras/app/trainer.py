# from abc import abstractmethod
from enum import Enum
import math

from .. import logging as Log
from ..base import Observable
from ..dataset import Dataset
from ..utils.progressbar import ProgressBar


class Event(Enum):
    TRAIN_BEGIN = 'train_begin'
    TRAIN_END = 'train_end'
    EPOCH_BEGIN = 'epoch_begin'
    EPOCH_END = 'epoch_end'
    EPOCH_TRAIN_BEGIN = 'epoch_train_begin'
    EPOCH_TRAIN_END = 'epoch_train_end'
    EPOCH_VALIDATION_BEGIN = 'epoch_validation_begin'
    EPOCH_VALIDATION_END = 'epoch_validation_end'
    BATCH_BEGIN = 'batch_begin'
    BATCH_END = 'batch_end'


# class Callback(object):
#
#     a


class Trainer(Observable):

    def __init__(self, optimizer, model, loss_func):
        super(Trainer, self).__init__()
        self._optimizer = optimizer
        self._model = model
        self._loss_func = loss_func
        self._initialize()

    def _initialize(self):
        def update(optimizer, loss):
            # suppose chainer API
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()
        self._update = update
        # self.add_hook(Event.TRAIN_BEGIN,
        #               lambda x: print(Event.TRAIN_BEGIN, x))
        # self._context

    def fit(self,
            x,
            y,
            batch_size=32,
            epochs=10,
            validation_data=None,
            verbose=True):

        train_dataset = Dataset(x, y)

        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'items, however it contains %d items' %
                                 len(validation_data))
            val_dataset = Dataset(val_x, val_y)
        else:
            do_validation = False

        if verbose:
            pbar = ProgressBar()
            self.add_hook(Event.EPOCH_TRAIN_BEGIN,
                          lambda data: pbar.start(data['size']))
            self.add_hook(Event.BATCH_BEGIN,
                          lambda data:
                          pbar.update((batch_size * data['batch_index']) + 1))
            self.add_hook(Event.EPOCH_TRAIN_END,
                          lambda data: pbar.finish())
            self.add_hook(Event.EPOCH_TRAIN_END,
                          lambda data: Log.i(
                              "[train] epoch {} - "
                              "#samples: {}, loss: {}"
                              .format(data['epoch'],
                                      data['size'],
                                      data['loss'])))
            if do_validation:
                self.add_hook(Event.EPOCH_VALIDATION_BEGIN,
                              lambda data: pbar.start(data['size']))
                self.add_hook(Event.EPOCH_VALIDATION_END,
                              lambda data: pbar.finish())
                self.add_hook(Event.EPOCH_VALIDATION_END,
                              lambda data: Log.i(
                                  "[validation] epoch {} - "
                                  "#samples: {}, loss: {}"
                                  .format(data['epoch'],
                                          data['size'],
                                          data['loss'])))

        self.notify(Event.TRAIN_BEGIN)

        forward = (self._model if callable(self._model)
                   else self._model.forward())
        lossfun = self._loss_func

        for epoch in range(1, epochs + 1):
            epoch_logs = {
                'epoch': epoch,
                'size': train_dataset.size,
            }
            epoch_logs['loss'] = []
            self.notify(Event.EPOCH_BEGIN, epoch_logs)

            self._process(forward, train_dataset, lossfun,
                          batch_size, epoch_logs)
            if do_validation:
                self._process(forward, val_dataset, lossfun,
                              batch_size, epoch_logs, train=False)

            self.notify(Event.EPOCH_END, epoch_logs)

        self.notify(Event.TRAIN_END)

        # return history

    def _process(self,
                 forward,
                 dataset,
                 lossfun,
                 batch_size,
                 logs={}, train=True):
        logs = logs.copy()
        logs['size'] = dataset.size
        num_batches = math.ceil(logs['size'] / batch_size)
        logs['num_batches'] = num_batches
        logs['loss'] = None
        self.notify(Event.EPOCH_TRAIN_BEGIN
                    if train else Event.EPOCH_VALIDATION_BEGIN, logs)
        logs['loss'] = 0.0
        for batch_index, batch in enumerate(
                dataset.batch(batch_size, colwise=True, shuffle=train)):
            xs, ts = batch[:-1], batch[-1]
            if len(xs) == 1:
                xs = xs[0]

            batch_logs = {
                'train': train,
                'batch_index': batch_index,
                'batch_size': len(ts),
                'xs': xs,
                'ts': ts,
                'ys': None,
                'loss': None,
            }
            self.notify(Event.BATCH_BEGIN, batch_logs)

            ys = forward(xs)
            loss = lossfun(ys, ts)

            batch_logs['ys'] = ys
            batch_logs['loss'] = loss
            logs['loss'] += loss

            if train:
                self._update(self._optimizer, loss)
            self.notify(Event.BATCH_END, batch_logs)
            del loss

        logs['loss'] /= num_batches
        self.notify(Event.EPOCH_TRAIN_END
                    if train else Event.EPOCH_VALIDATION_END, logs)
