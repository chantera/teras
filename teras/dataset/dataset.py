from collections.abc import Sequence

import numpy as np


def _get_row_iterator(samples, batches):
    for indices in batches:
        yield np.take(samples, indices, axis=0)


def _get_col_iterator(columns, batches):
    for indices in batches:
        yield tuple(_take(column, indices) for column in columns)


def _take(column, indices):
    if isinstance(column, (list, tuple)) \
            and isinstance(column[0], np.ndarray):
        return type(column)(column[idx] for idx in indices)
    else:
        return np.take(column, indices, axis=0)


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
                    if isinstance(column[0], np.ndarray)
                    and column[0].ndim <= 1 else column
                    for column in map(tuple, zip(*samples[0])))
                self._columns = columns
                self._n_cols = len(columns)
            else:
                self._columns = (samples[0],)
        elif self._n_cols > 1:
            self._columns = samples
        self._samples = list(map(tuple, zip(*self._columns)))
        self._len = len(self._samples)

    def __iter__(self):
        return (sample for sample in self._samples)

    def batch(self, size, shuffle=False, colwise=True):
        indices = np.arange(self._len)
        if shuffle:
            np.random.shuffle(indices)
        batches = [indices[i: i + size]
                   for i in range(0, self._len, size)]
        if colwise:
            return _get_col_iterator(self._columns, batches)
        else:
            return _get_row_iterator(self._samples, batches)

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


class BucketDataset(Dataset):

    def __init__(self, *samples, key=0, equalize_by_key=False):
        super().__init__(*samples)
        self._key = key
        self._equalize = equalize_by_key

    def batch(self, size, shuffle=False, colwise=True):
        key = self._key
        lengths = ((float(len(sample[key])), index)
                   for index, sample in enumerate(self))
        if shuffle:
            # Add noise to the lengths so that the sorting isn't deterministic.
            lengths = ((length + np.random.random(), index)
                       for length, index in lengths)
            # Randomly choose the order from (ASC, DESC).
            lengths = sorted(lengths, key=lambda x: x[0],
                             reverse=np.random.random() > .5)
        # Bucketing
        if self._equalize:
            buckets = []
            bucket = []
            accum_length = 0
            for length, index in lengths:
                if accum_length + int(length) > size:
                    buckets.append(np.array(bucket))
                    bucket = []
                    accum_length = 0
                bucket.append(index)
                accum_length += int(length)
            if bucket:
                buckets.append(np.array(bucket))
        else:
            lengths = list(lengths)
            buckets = [np.array([index for _, index in lengths[i: i + size]])
                       for i in range(0, self._len, size)]
        if shuffle:
            np.random.shuffle(buckets)
        if colwise:
            return _get_col_iterator(self._columns, buckets)
        else:
            return _get_row_iterator(self._samples, buckets)
