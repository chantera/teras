from teras.dataset import Dataset
from teras.training import listeners
from teras.training.event import Dispatcher, TrainEvent
from teras.utils import logging
from teras.utils.collections import PseudoImmutableMap


class Trainer(Dispatcher):
    EventClass = TrainEvent

    def __init__(self, optimizer, forward, loss_func, accuracy_func=None):
        super(Trainer, self).__init__()
        self._optimizer = optimizer
        self._forward = forward
        self._loss_func = loss_func
        self._acc_func = accuracy_func
        self._initialize()

    def _initialize(self):
        def update(optimizer, loss):
            # suppose chainer API
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()
        self._update = update
        self._converter = None
        self._reporter = None

    def configure(self, config=None, **kwargs):
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            raise TypeError("`config` must be a dict")
        config.update(kwargs)
        if 'update' in config:
            self._update = config['update']
        if 'hooks' in config:
            for event, hook in config['hooks'].items():
                self.add_hook(event, hook)
        if 'listeners' in config:
            for listener in config['listeners']:
                self.add_listener(listener)
        if 'converter' in config:
            self._converter = config['converter']

    def fit(self, data, valid_data=None, epochs=10, batch_size=32):
        if isinstance(data, Dataset):
            train_dataset = data
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            train_dataset = Dataset(*data)
        else:
            raise ValueError('invalid data: {}'.format(type(data)))

        if valid_data:
            do_validation = True
            if isinstance(valid_data, Dataset):
                val_dataset = valid_data
            elif isinstance(valid_data, (tuple, list)) \
                    and len(valid_data) == 2:
                val_dataset = Dataset(*valid_data)
            else:
                raise ValueError('When passing valid_data, '
                                 'it must be dataset or contain '
                                 'two (x_val, y_val) items: {}'
                                 .format(type(valid_data)))
        else:
            do_validation = False

        self._reporter = listeners.Reporter(logging.getLogger())
        self.add_listener(self._reporter, priority=110)
        if self._acc_func is not None:
            def _report_accuracy(data):
                listeners.report(
                    {"accuracy": self._acc_func(data['ys'], data['ts'])})
            self.add_hook(TrainEvent.BATCH_END, _report_accuracy, priority=120)

        forward = self._forward
        if not callable(forward):
            if hasattr(self._forward, 'forward'):
                forward = self._forward.forward
            else:
                raise RuntimeError('`forward` is not callable')
        lossfun = self._loss_func
        convert = (self._converter if callable(self._converter)
                   else lambda x: x)

        history = []

        self.notify(TrainEvent.TRAIN_BEGIN)

        def main_loop():
            for epoch in range(1, epochs + 1):
                epoch_logs = PseudoImmutableMap(
                    epoch=epoch,
                    size=train_dataset.size,
                )
                self.notify(TrainEvent.EPOCH_BEGIN, epoch_logs)

                self._process(forward, train_dataset, lossfun, convert,
                              batch_size, epoch_logs.copy(), train=True)
                if do_validation:
                    self._process(forward, val_dataset, lossfun, convert,
                                  batch_size, epoch_logs.copy(), train=False)

                self.notify(TrainEvent.EPOCH_END, epoch_logs)

        if self._reporter is not None:
            with self._reporter:
                main_loop()
        else:
            main_loop()

        self.notify(TrainEvent.TRAIN_END)

        return history

    def _process(self,
                 forward,
                 dataset,
                 lossfun,
                 convert,
                 batch_size,
                 logs=None, train=True):
        if logs is None:
            logs = PseudoImmutableMap()
        logs.data['size'] = dataset.size
        iterator = dataset.batch(batch_size, colwise=True, shuffle=train)
        num_batches = len(iterator)
        logs.data['num_batches'] = num_batches
        logs.data['loss'] = None
        self.notify(TrainEvent.EPOCH_TRAIN_BEGIN
                    if train else TrainEvent.EPOCH_VALIDATE_BEGIN, logs)
        logs.data['loss'] = 0.0
        for batch_index, batch in enumerate(iterator):
            xs, ts = batch[:-1], batch[-1]
            if len(xs) == 1:
                xs = [convert(xs[0])]
            else:
                xs = convert(xs)
            ts = convert(ts)

            batch_logs = PseudoImmutableMap(
                train=train,
                batch_index=batch_index,
                batch_size=len(ts),
                xs=xs,
                ts=ts,
                ys=None,
                loss=None,
                num_batches=num_batches,
            )
            self.notify(TrainEvent.BATCH_BEGIN, batch_logs)

            ys = forward(*batch_logs['xs'])
            loss = lossfun(ys, batch_logs['ts'])

            batch_logs.data['ys'] = ys
            batch_logs.data['loss'] = loss.__float__()
            logs.data['loss'] += loss.__float__()

            if train:
                self._update(self._optimizer, loss)
            self.notify(TrainEvent.BATCH_END, batch_logs)
            del loss

        logs.data['loss'] /= num_batches
        self.notify(TrainEvent.EPOCH_TRAIN_END
                    if train else TrainEvent.EPOCH_VALIDATE_END, logs)
