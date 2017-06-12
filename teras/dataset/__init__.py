from collections.abc import Iterable, Iterator, Sequence
from operator import itemgetter

import numpy as np


class Dataset(Sequence):
    """Immutable Class"""

    def __init__(self, *samples):
        self._n_cols = len(samples)
        if self._n_cols == 0:
            self._samples = []
            self._n_cols = 1
            self._dtype = list
        elif self._n_cols == 1:
            self._dtype = type(samples[0])
            if self._dtype is np.ndarray:
                self._samples = samples[0]
                self._n_cols = self._samples.shape[0]
            else:
                self._samples = list(samples[0])
                first_sample = self._samples[0]
                if (len(self._samples) > 0
                        and (type(first_sample) is tuple
                             or type(first_sample) is list)):
                    self._n_cols = len(first_sample)
                    self._dtype = (np.ndarray
                                   if type(first_sample[0]) is np.ndarray
                                   else type(first_sample))
        elif self._n_cols > 1:
            self._dtype = type(samples[0])
            if self._dtype is np.ndarray:
                self._samples = [_samples for _samples in zip(*samples)]
            else:
                self._samples \
                    = [self._dtype(_samples) for _samples in zip(*samples)]
        self._len = len(self._samples)
        self._indexes = np.arange(self._len)

    def batch(self, size, shuffle=False, colwise=False):
        if shuffle:
            np.random.shuffle(self._indexes)
        return _DatasetBatchIterator(
            Dataset(self.take(self._indexes)), size, colwise)

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if self._dtype == tuple:
            return tuple(self._samples)[key]
        return self._samples[key]

    def take(self, indices):
        if type(self._samples) is np.ndarray:
            return self._samples.take(indices, axis=0)
        elif isinstance(indices, Iterable):
            _type = tuple if self._dtype is np.ndarray else self._dtype
            return _type(itemgetter(*indices)(self._samples))
        return self._samples[indices]

    def __repr__(self):
        return repr(self._samples)

    def __str__(self):
        return str(self._samples)

    def cols(self):
        if self._dtype is np.ndarray:
            if self._n_cols > 1:
                return np.swapaxes(self._samples, 0, 1)
            return self._samples
        else:
            if self._n_cols > 1:
                return tuple(self._dtype(col) for col in zip(*self._samples))
            return self._dtype(self._samples),

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self.__len__()

    @property
    def n_cols(self):
        return self._n_cols


class _DatasetBatchIterator(Iterator):

    def __init__(self, dataset, batch_size, colwise=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._colwise = colwise

    def __iter__(self):
        dataset = self._dataset
        container_type = type(dataset._samples)
        dtype = dataset.dtype
        if self._colwise:
            if container_type is np.ndarray:
                def _take(dataset, offset, batch_size):
                    return dataset.cols()[:, offset:offset + batch_size]
            elif dataset.n_cols == 1:
                def _take(dataset, offset, batch_size):
                    return dataset[offset:offset + batch_size],
            else:
                def _take(dataset, offset, batch_size):
                    _type = np.array if dtype is np.ndarray else dtype
                    return tuple(_type(col) for col in zip(
                        *dataset._samples[offset:offset + batch_size]))
        else:
            def _take(dataset, offset, batch_size):
                return dataset[offset:offset + batch_size]
        size = dataset.size
        offset = 0
        while True:
            if offset >= size:
                raise StopIteration()
            yield _take(dataset, offset, self._batch_size)
            offset += self._batch_size

    def __next__(self):
        self.__iter__()
