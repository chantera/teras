from collections.abc import Sequence

import numpy as np

from teras.dataset.mnist import get_mnist  # NOQA


class Dataset(Sequence):
    """Immutable Class"""

    def __init__(self, *samples):
        self._n_cols = len(samples)
        if self._n_cols == 0:
            self._columns = ([],)
            self._n_cols = 1
        elif self._n_cols == 1:
            if isinstance(samples[0], (list, tuple)):
                columns = tuple(
                    np.array(column)
                    if isinstance(column[0], np.ndarray) else column
                    for column in map(tuple, zip(*samples[0])))
                self._columns = columns
                self._n_cols = len(columns)
            else:
                self._columns = (samples[0],)
        elif self._n_cols > 1:
            self._columns = samples
        self._samples = list(map(tuple, zip(*self._columns)))
        self._len = len(self._samples)
        self._indices = np.arange(self._len)

    def __iter__(self):
        return (sample for sample in self._samples)

    def batch(self, size, shuffle=False, colwise=True):
        if shuffle:
            np.random.shuffle(self._indices)
        if colwise:
            return self._get_col_iterator(size)
        else:
            return self._get_row_iterator(size)

    def _get_row_iterator(self, batch_size):
        size = len(self)
        offset = 0
        while True:
            if offset >= size:
                raise StopIteration()
            indices = self._indices[offset:offset + batch_size]
            yield np.take(self._samples, indices, axis=0)
            offset += batch_size

    def _get_col_iterator(self, batch_size):
        size = len(self)
        offset = 0
        while True:
            if offset >= size:
                raise StopIteration()
            indices = self._indices[offset:offset + batch_size]
            yield tuple(np.take(column, indices, axis=0)
                        for column in self._columns)
            offset += batch_size

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        return self._samples.__getitem__(key)

    def take(self, indices):
        return np.take(self._samples, indices, axis=0)

    def __reversed__(self):
        return self._samples.__reversed__()

    def __contains(self, sample):
        return self._samples.__contains__(sample)

    def index(self, sample):
        return self._samples.index(sample)

    def count(self, sample):
        return self._samples.count(sample)

    def __repr__(self):
        return repr(self._samples)

    def __str__(self):
        return str(self._samples)

    def cols(self):
        return self._columns

    @property
    def size(self):
        return self.__len__()

    @property
    def n_cols(self):
        return self._n_cols
