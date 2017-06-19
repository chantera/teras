from ..base.event import Callback
from .event import TrainEvent
from ..utils.progressbar import ProgressBar
from .. import logging as Log


class ProgressCallback(Callback):

    def __init__(self, name="progress_callback", **kwargs):
        super(ProgressCallback, self).__init__(name, **kwargs)
        self._pbar = ProgressBar()
        self.implement(TrainEvent.EPOCH_TRAIN_BEGIN, self.init_progressbar)
        self.implement(TrainEvent.BATCH_BEGIN, self.update_progressbar)
        self.implement(TrainEvent.EPOCH_TRAIN_END, self.finish_progressbar)

    def init_progressbar(self, data):
        self._pbar.start(data['size'])

    def update_progressbar(self, data):
        self._pbar.update((data['batch_size'] * data['batch_index']) + 1)

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
            'loss': float(data['loss'])
        }
        Log.i("[training] epoch {} - "
              "#samples: {}, loss: {:.8f}, accuracy: {:.8f}"
              .format(data['epoch'], data['size'],
                      metrics['loss'], metrics['accuracy']))
        self._history.append({'training': metrics, 'validation': None})

    def on_epoch_validate_end(self, data):
        metrics = {
            'accuracy': self._logs['accuracy'] / data['num_batches'],
            'loss': float(data['loss'])
        }
        Log.i("[validation] epoch {} - "
              "#samples: {}, loss: {:.8f}, accuracy: {:.8f}"
              .format(data['epoch'], data['size'],
                      metrics['loss'], metrics['accuracy']))
        self._history[-1]['validation'] = metrics
