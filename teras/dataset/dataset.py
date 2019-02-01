from collections import defaultdict
from collections.abc import Sequence

import numpy as np


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
                return
            indices = self._indices[offset:offset + batch_size]
            yield np.take(self._samples, indices, axis=0)
            offset += batch_size

    def _get_col_iterator(self, batch_size):
        size = len(self)
        offset = 0

        def _take(column, indices):
            if isinstance(column, (list, tuple)) \
                    and isinstance(column[0], np.ndarray):
                return type(column)(column[idx] for idx in indices)
            else:
                return np.take(column, indices, axis=0)

        while True:
            if offset >= size:
                return
            indices = self._indices[offset:offset + batch_size]
            yield tuple(_take(column, indices) for column in self._columns)
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
            for item in lengths:
                bucket.append(item)
                accum_length += int(item[1])
                if accum_length >= size:
                    buckets.append(bucket)
                    bucket = []
            if bucket:
                buckets.append(bucket)
        else:
            lengths = list(lengths)
            buckets = [lengths[i: i + size]
                       for i in range(0, self._len, size)]
        if shuffle:
            np.random.shuffle(buckets)
        if colwise:
            return self._get_col_iterator(buckets)
        else:
            return self._get_row_iterator(buckets)

    def _get_row_iterator(self, buckets):
        iterator = iter(buckets)
        while True:
            try:
                bucket = next(iterator)
            except StopIteration:
                return
            indices = [item[1] for item in bucket]
            yield np.take(self._samples, indices, axis=0)

    def _get_col_iterator(self, buckets):
        iterator = iter(buckets)
        while True:
            try:
                bucket = next(iterator)
            except StopIteration:
                return
            indices = [item[1] for item in bucket]
            yield tuple(np.take(column, indices, axis=0)
                        for column in self._columns)


class GroupedDataset(Dataset):

    def make_groups(self, batch_size, column=0):
        self.make_groups_into(int(np.ceil(len(self) / batch_size)))

    def make_groups_into(self, size, column=0):
        indices = defaultdict(list)
        for index, sample in enumerate(self):
            length = len(sample[column])
            indices[length].append(index)
        groups = []
        n_samples_per_group = int(np.ceil(len(self) / size))
        group, count = [], 0
        for length, samples in sorted(indices.items()):
            for sample in samples:
                group.append(sample)
                count += 1
                if count == n_samples_per_group:
                    groups.append(group)
                    group, count = [], 0
        if count > 0:
            groups.append(group)
        self._groups = groups
        assert len(groups) == size
        self._group_indices = np.arange(size)

    def batch(self, size=-1, shuffle=False, colwise=True):
        if shuffle:
            np.random.shuffle(self._group_indices)
        if colwise:
            return self._get_col_iterator()
        else:
            return self._get_row_iterator()

    def _get_row_iterator(self):
        iterator = iter(self._group_indices)
        while True:
            index = next(iterator)
            indices = self._groups[index]
            yield np.take(self._samples, indices, axis=0)

    def _get_col_iterator(self):
        iterator = iter(self._group_indices)
        while True:
            index = next(iterator)
            indices = self._groups[index]
            yield tuple(np.take(column, indices, axis=0)
                        for column in self._columns)
