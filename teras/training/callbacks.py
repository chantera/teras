import os

from teras import logging as Log
from teras.base.event import Callback
from teras.training.event import TrainEvent
from teras.utils.progressbar import ProgressBar
import teras.utils


class ProgressCallback(Callback):

    def __init__(self, name="progress_callback", **kwargs):
        super(ProgressCallback, self).__init__(name, **kwargs)
        self._pbar = ProgressBar()
        self.implement(TrainEvent.EPOCH_TRAIN_BEGIN, self.init_progressbar)
        self.implement(TrainEvent.BATCH_END, self.update_progressbar)
        self.implement(TrainEvent.EPOCH_TRAIN_END, self.finish_progressbar)

    def init_progressbar(self, data):
        self._pbar.start(data['size'])
        self._count = 0

    def update_progressbar(self, data):
        self._count += data['batch_size']
        self._pbar.update(self._count)

    def finish_progressbar(self, data):
        self._pbar.finish()


class Reporter(Callback):

    def __init__(self, accuracy_func, name="reporter", **kwargs):
        super(Reporter, self).__init__(name, **kwargs)
        self._acc_func = accuracy_func
        self._logs = {}
        self._history = []

    def get_history(self):
        return self._history

    def on_train_begin(self, data):
        self._history = []

    def on_epoch_train_begin(self, data):
        self._logs = {
            'accuracy': 0.0,
            'loss': 0.0,
        }

    on_epoch_validate_begin = on_epoch_train_begin

    def on_batch_end(self, data):
        accuracy = self._acc_func(data['ys'], data['ts'])
        self._logs['accuracy'] += float(accuracy)

    def on_epoch_train_end(self, data):
        metrics = {
            'accuracy': self._logs['accuracy'] / data['num_batches'],
            'loss': data['loss']
        }
        Log.i("[training] epoch {} - "
              "#samples: {}, loss: {:.8f}, accuracy: {:.8f}"
              .format(data['epoch'], data['size'],
                      metrics['loss'], metrics['accuracy']))
        self._history.append({'training': metrics, 'validation': None})

    def on_epoch_validate_end(self, data):
        metrics = {
            'accuracy': self._logs['accuracy'] / data['num_batches'],
            'loss': data['loss']
        }
        Log.i("[validation] epoch {} - "
              "#samples: {}, loss: {:.8f}, accuracy: {:.8f}"
              .format(data['epoch'], data['size'],
                      metrics['loss'], metrics['accuracy']))
        self._history[-1]['validation'] = metrics


class Saver(Callback):

    def __init__(self, model, basename, directory='', context=None,
                 interval=1, name="saver", **kwargs):
        super(Saver, self).__init__(name, **kwargs)
        self._model = model
        self._basename = os.path.join(os.path.expanduser(directory), basename)
        self._context = context
        if not isinstance(interval, int):
            raise ValueError("interval must be specified as int value: "
                             "actual('{}')".format(type(interval).__name__))
        self._interval = interval

    def on_train_begin(self, data):
        if self._context is not None:
            context_file = self._basename + '.context'
            Log.i("saving the context to {} ...".format(context_file))
            with open(context_file, 'wb') as f:
                teras.utils.dump(self._context, f)

    def on_epoch_end(self, data):
        epoch = data['epoch']
        if epoch % self._interval == 0:
            model_file = "{}.{}.pkl".format(self._basename, epoch)
            Log.i("saving the model to {} ...".format(model_file))
            with open(model_file, 'wb') as f:
                teras.utils.dump(self._model, f)
