from collections import OrderedDict
import os
import pickle
import logging

import numpy as np
from teras.training.event import Listener
from teras.utils.collections import ImmutableMap


class ProgressBar(Listener):
    """
    Example::

        >>> from tqdm import tqdm
        >>> import time
        >>> pbar = ProgressBar(lambda n: tqdm(total=n))
        >>> pbar.init(512)
        >>> for _ in range(16):
        >>>     time.sleep(0.1)
        >>>     pbar.update(32)
        >>> pbar.close()
    """
    name = "progressbar"

    def __init__(self, factory, **kwargs):
        super().__init__(**kwargs)
        self._pbar = None
        self._factory = factory

    def init(self, total):
        self.close()
        self._pbar = self._factory(total)

    def update(self, n):
        self._pbar.update(n)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def __del__(self):
        self.close()

    def on_epoch_train_begin(self, data):
        self.init(data['size'])

    def on_batch_end(self, data):
        self.update(data['batch_size'])

    def on_epoch_train_end(self, data):
        self.close()

    on_epoch_validate_begin = on_epoch_train_begin

    on_epoch_validate_end = on_epoch_train_end


_reporters = []


def report(values):
    if not _reporters:
        return
    for reporter in reversed(_reporters):
        reporter.report(values)


class Reporter(Listener):
    name = "reporter"

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self._logger = logger
        self._logs = OrderedDict()
        self._reported = 0
        self._history = []

    def __enter__(self):
        _reporters.append(self)

    def __exit__(self, exc_type, exc_value, traceback):
        _reporters.pop()

    def report(self, values):
        for name, value in values.items():
            if name != "loss" and "loss" in name:
                loss = self._logs.get(name, [])
                loss.append(float(value))
                values[name] = loss
            elif "accuracy" in name:
                accuracy = self._logs.get(name, 0.0)
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    if isinstance(accuracy, float):
                        accuracy = [0, 0]
                    accuracy[0] += value[0]
                    accuracy[1] += value[1]
                else:
                    accuracy += float(value)
                values[name] = accuracy
        self._logs.update(values)
        self._reported += 1

    def get_summary(self):
        summary = OrderedDict()
        for name, value in self._logs.items():
            if name != "loss" and "loss" in name:
                n = len(value)
                summary[name] = sum(value) / n if n > 0 else np.nan
            elif "accuracy" in name:
                if isinstance(value, list):
                    correct, total = value[:2]
                    if total == 0:
                        accuracy = np.nan
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
        self._logs.clear()
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
        self._logger.info(message)
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
            self._logger.info(", ".join(message))


class Saver(Listener):
    name = "saver"

    class Context(ImmutableMap):
        def __getattr__(self, name):
            if name in self.data:
                return self.data[name]
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(type(self).__name__, name))

        def __hash__(self):
            return hash(tuple(sorted(self.data.items())))

    def __init__(self, model, basename, directory='', context=None, interval=1,
                 save_from=None, save_best=False, evaluate=None,
                 serializer=None, logger=None, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._basename = os.path.join(os.path.expanduser(directory), basename)
        if context is not None and not isinstance(context, Saver.Context):
            context = Saver.Context(context)
        self._context = context
        if not isinstance(interval, int):
            raise ValueError("interval must be specified as int value: "
                             "actual('{}')".format(type(interval).__name__))
        self._interval = interval
        self._save_from = save_from
        self._save_best = save_best
        self._evaluate = evaluate
        self._best_value = -float('inf')
        self._serializer = serializer if serializer is not None else pickle
        self._logger = logger \
            if logger is not None else logging.getLogger(__name__)

    def save_context(self, context):
        if not isinstance(context, Saver.Context):
            raise TypeError('`context` must be a Saver.Context object')
        file = self._basename + '.context'
        self._logger.info("saving the context to {} ...".format(file))
        with open(file, 'wb') as f:
            self._serializer.dump(context, f)

    def save_model(self, model, suffix=''):
        file = "{}{}.pkl".format(self._basename, suffix)
        self._logger.info("saving the model to {} ...".format(file))
        with open(file, 'wb') as f:
            self._serializer.dump(model, f)

    @staticmethod
    def load_context(model_file, deserializer=None):
        if deserializer is None:
            deserializer = pickle
        _dir, _file = os.path.split(model_file)
        context_file = os.path.basename(_file).split('.')[0] + '.context'
        context_file = os.path.join(_dir, context_file)
        with open(context_file, 'rb') as f:
            context = deserializer.load(f)
        return context

    def on_train_begin(self, data):
        if self._context is not None:
            self.save_context(self._context)

    def on_epoch_validate_end(self, data):
        if self._save_best:
            self._trigger_save(data)

    def on_epoch_end(self, data):
        if not self._save_best:
            self._trigger_save(data)

    def _trigger_save(self, data):
        epoch = data['epoch']
        if self._save_from is not None and epoch < self._save_from:
            return
        if epoch % self._interval == 0:
            if self._save_best:
                if callable(self._evaluate):
                    value = self._evaluate(data)
                else:
                    value = -data['loss']
                if value <= self._best_value:
                    return
                self._logger.info("update the best score - new: {}, old: {}"
                                  .format(value, self._best_value))
                self._best_value = value
                self.save_model(self._model)
            else:
                self.save_model(self._model, suffix='.' + str(epoch))
