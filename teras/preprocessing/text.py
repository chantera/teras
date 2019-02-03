from collections import Iterable, UserDict
import copy
import re

import numpy as np


class Dict(UserDict):

    def __init__(self):
        super().__init__()
        self._index = -1
        self._id2word = {}

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def add(self, key):
        return self[key]

    def __missing__(self, key):
        self._index = idx = self._index + 1
        self.data[key] = idx
        self._id2word[idx] = key
        return idx

    def __setitem__(self, key, item):
        if not isinstance(item, int):
            raise ValueError("item must be int, but {} given"
                             .format(type(item)))
        if self._id2word.get(item, key) != key:
            raise ValueError("item has already been assigned "
                             "to another key".format(type(item)))
        if key in self.data:
            del self._id2word[self.data[key]]
        self.data[key] = item
        self._id2word[item] = key
        if item > self._index:
            self._index = item

    def __delitem__(self, key):
        del self._id2word[self.data.pop(key)]

    def copy(self):
        data = self.data
        id2word = self._id2word
        try:
            self.data = {}
            self._id2word = {}
            c = copy.copy(self)
        finally:
            self.data = data
            self._id2word = id2word
        c.update(data)
        return c

    @classmethod
    def fromkeys(cls, iterable):
        self = cls()
        for idx, key in enumerate(iterable):
            self.data[key] = idx
            self._id2word[idx] = key
        self._index = idx
        return self

    def lookup(self, value):
        return self._id2word[value]

    @property
    def size(self):
        return self._index + 1

    __marker = object()

    def pop(self, key, default=__marker):
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default

    def popitem(self):
        result = self.data.popitem()
        del self._id2word[result[1]]
        return result

    def clear(self):
        self.data.clear()
        self._id2word.clear()
        self._index = -1

    def update(self, *args, **kwargs):
        d = dict()
        d.update(*args, **kwargs)
        for k, v in d.items():
            self[k] = v

    def setdefault(self, key, default=-1):
        if key in self:
            return self[key]
        self[key] = default
        return default


class Vocab(object):

    def __init__(self, unknown="<UNK>"):
        self._dict = Dict()
        self._unknown_id = self._dict[unknown]

    def add(self, word):
        return self._dict.add(word)

    def __getitem__(self, word):
        return self._dict[word] if word in self._dict else self._unknown_id

    def __contains__(self, word):
        return word in self._dict

    def __len__(self):
        """including the unknown word"""
        return self._dict.size

    def lookup(self, id):
        return self._dict.lookup(id)

    def __repr__(self):
        return repr(self._dict)

    @property
    def unknown(self):
        return self._unknown_id

    @classmethod
    def from_words(cls, iterable, unknown="<UNK>"):
        v = cls(unknown)
        for word in iterable:
            v.add(word)
        return v


class EmbeddingVocab(Vocab):

    def __init__(self, unknown="<UNK>", file=None, dim=50, dtype=np.float32,
                 initializer=None, serialize_embeddings=False):
        if file is not None:
            embed_file, vocab_file = file, None
            if isinstance(file, (list, tuple)):
                assert len(file) == 2
                embed_file, vocab_file = file
            vdict, embeddings = load_embeddings(embed_file, vocab_file, dtype)
        else:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("embed_size must be a positive integer value")
            vdict, embeddings = Dict(), np.empty((0, dim), dtype)

        if initializer is None or initializer == 'normal':
            initializer = EmbeddingVocab.random_normal
        elif initializer == 'uniform':
            initializer = EmbeddingVocab.random_uniform
        elif not callable(initializer):
            raise ValueError("invalid initializer")

        self._dict = vdict
        self._unknown_id = self._dict[unknown]
        self._file = file
        self._embeddings = embeddings
        self._initializer = initializer
        self._enable_serialize = serialize_embeddings

    def get_embeddings(self, normalize=False):
        if self._embeddings is None:
            raise RuntimeError("cannot retrieve embeddings")
        n_elements, dim = self._embeddings.shape
        n_uninitialized = len(self) - n_elements
        if n_uninitialized > 0:
            new_vectors = self._initializer(
                (n_uninitialized, dim), self._embeddings.dtype)
            self._embeddings = np.r_[self._embeddings, new_vectors]
        return self.normalize(self._embeddings, normalize) \
            if normalize else self._embeddings

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self._enable_serialize:
            state['_embeddings'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_words(cls, iterable, unknown="<UNK>", **kwargs):
        v = cls(unknown, **kwargs)
        for word in iterable:
            v.add(word)
        return v

    @staticmethod
    def random_normal(shape, dtype=np.float32):
        return np.random.normal(0, 1, shape).astype(dtype, copy=False)

    @staticmethod
    def random_uniform(shape, dtype=np.float32):
        return np.random.uniform(-1, 1, shape).astype(dtype, copy=False)

    @staticmethod
    def normalize(embeddings, method='l2'):
        if method == 'l2':
            l2 = np.linalg.norm(embeddings, axis=1, keepdims=True)
            l2[l2 == 0] = 1
            embeddings = embeddings / l2
        elif method == 'zscore':
            mean = np.mean(embeddings, axis=1, keepdims=True)
            std = np.std(embeddings, axis=1, keepdims=True)
            embeddings = (embeddings - mean) / std
        else:
            raise ValueError('unsupported method was specified: {}'
                             .format(method))
        return embeddings


def load_embeddings(embed_file, vocab_file=None, dtype=np.float32):
    vocabulary = Dict()
    embeddings = []
    if vocab_file:
        with open(embed_file) as ef, open(vocab_file) as vf:
            for line1, line2 in zip(ef, vf):
                word = line2.strip()
                vector = line1.strip().split()
                if word not in vocabulary:
                    vocabulary.add(word)
                    embeddings.append(np.array(vector, dtype=dtype))
    else:
        with open(embed_file) as f:
            lines = f.readlines()
            index = 0
            if len(lines[0].strip().split()) <= 2:
                index = 1  # skip header
            for line in lines[index:]:
                cols = line.strip().split()
                word = cols[0]
                if word not in vocabulary:
                    vocabulary.add(word)
                    embeddings.append(np.array(cols[1:], dtype=dtype))
    return vocabulary, np.vstack(embeddings)


def split(sentence):
    return sentence.split()


def lower(word):
    return str(word).lower()


def replace_number(word):
    return re.sub(r'^\d+(,\d+)*(\.\d+)?$', '<NUM>', word.lower())


def raw(word):
    return word


class Preprocessor(object):

    def __init__(self, vocab=None,
                 tokenizer=split,
                 preprocess=lower,
                 pad=None,
                 dtype=np.int32):
        self.vocab = vocab if vocab is not None else Vocab()
        self.tokenizer = tokenizer
        self.preprocess = preprocess if preprocess else raw
        self.pad_id = self.vocab[pad] if pad is not None else -1
        self._dtype = dtype

    def fit(self, document):
        for token in self.tokenize(document):
            self.vocab.add(token)
        return self

    def transform(self, document, length=None):
        return self.fit_transform(document, length, False)

    def fit_transform(self, document, length=None, fit=True):
        if (callable(fit) and fit()) or fit:
            _convert = self.vocab.add
        else:
            _convert = self.vocab.__getitem__
        ids = np.array([_convert(token) for token
                        in self.tokenize(document)], self._dtype)
        return self.pad(ids, length) if length is not None else ids

    def tokenize(self, document):
        if isinstance(document, str):
            tokens = self.tokenizer(document)
        elif isinstance(document, Iterable):
            tokens = document
        else:
            raise ValueError(
                'document must be an instance of str or Iterable')
        return (self.preprocess(token) for token in tokens)

    def pad(self, ids, length):
        n = len(ids)
        if length - n < 0:
            raise ValueError("ids length exceeds the specified length value")
        pads = np.full(length, self.pad_id, self._dtype)
        pads[:n] = ids[:]
        return pads
