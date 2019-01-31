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


_reporters = []


def report(values):
    if _reporters:
        _reporters[-1].report(values)


class Reporter(Callback):

    def __init__(self, name="reporter", **kwargs):
        super(Reporter, self).__init__(name, **kwargs)
        self._logs = {}
        self._reported = 0
        self._history = []

    def __enter__(self):
        _reporters.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _reporters.pop()

    def report(self, values):
        for name, value in values.items():
            if "accuracy" in name:
                accuracy = self._logs.get(name, 0.0)
                if hasattr(value, "__len__") and len(value) == 2:
                    if isinstance(accuracy, float):
                        accuracy = [0, 0]
                    accuracy[0] += value[0]
                    accuracy[1] += value[1]
                else:
                    accuracy += float(accuracy)
                values[name] = accuracy
        self._logs.update(values)
        self._reported += 1

    def get_summary(self):
        summary = {}
        for name, value in self._logs.items():
            if "accuracy" in name:
                if isinstance(value, list):
                    correct, total = value[:2]
                    if total == 0:
                        import numpy
                        accuracy = numpy.nan
                    else:
                        accuracy = correct / total
                else:
                    accuracy = value / self._reported
                summary[name] = accuracy
            else:
                summary[name] = value
        return summary

    def get_history(self):
        return self._history

    def on_train_begin(self, data):
        self._history = []

    def on_epoch_train_begin(self, data):
        self._logs = {}
        self._reported = 0

    on_epoch_validate_begin = on_epoch_train_begin

    def on_epoch_train_end(self, data):
        self.report({'loss': data['loss']})
        summary = self.get_summary()
        self._output_log("training", summary, data)
        self._history.append({'training': summary, 'validation': None})

    def on_epoch_validate_end(self, data):
        self.report({'loss': data['loss']})
        summary = self.get_summary()
        self._output_log("validation", summary, data)
        self._history[-1]['validation'] = summary

    def _output_log(self, label, summary, data):
        message = "[{}] epoch {} - #samples: {}, loss: {:.8f}".format(
            label, data['epoch'], data['size'], summary['loss'])
        if 'accuracy' in summary:
            message += ", accuracy: {:.8f}".format(summary['accuracy'])
            v = self._logs.get('accuracy', None)
            if isinstance(v, list) and v[1] > 0:
                message += " ({}/{})".format(v[0], v[1])
        Log.i(message)
        message = []
        for name, value in summary.items():
            if name == 'loss' or name == 'accuracy':
                continue
            if isinstance(value, float):
                message.append("{}: {:.8f}".format(name, value))
            else:
                message.append("{}: {}".format(name, value))
            if 'accuracy' in name:
                v = self._logs.get(name, None)
                if isinstance(v, list) and v[1] > 0:
                    message[-1] += " ({}/{})".format(v[0], v[1])
        if message:
            Log.i(", ".join(message))


class Saver(Callback):

    def __init__(self, model, basename, directory='', context=None,
                 interval=1, save_from=None, name="saver", **kwargs):
        super(Saver, self).__init__(name, **kwargs)
        self._model = model
        self._basename = os.path.join(os.path.expanduser(directory), basename)
        self._context = context
        if not isinstance(interval, int):
            raise ValueError("interval must be specified as int value: "
                             "actual('{}')".format(type(interval).__name__))
        self._interval = interval
        self._save_from = save_from

    def on_train_begin(self, data):
        if self._context is not None:
            context_file = self._basename + '.context'
            Log.i("saving the context to {} ...".format(context_file))
            with open(context_file, 'wb') as f:
                teras.utils.dump(self._context, f)

    def on_epoch_end(self, data):
        epoch = data['epoch']
        if self._save_from is not None and data['epoch'] < self._save_from:
            return
        if epoch % self._interval == 0:
            model_file = "{}.{}.pkl".format(self._basename, epoch)
            Log.i("saving the model to {} ...".format(model_file))
            with open(model_file, 'wb') as f:
                teras.utils.dump(self._model, f)
