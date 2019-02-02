from collections import Iterable, UserDict
import copy
import re
import warnings

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


def split(sentence):
    return sentence.split()


def lower(word):
    return str(word).lower()


def replace_number(word):
    return re.sub(r'^\d+(,\d+)*(\.\d+)?$', '<NUM>', word.lower())


def raw(word):
    return word


def _np_pad(x, length, value, dtype=np.int32):
    y = np.full(length, value, dtype)
    y[:len(x)] = x[:]
    return y


class Preprocessor(object):

    def __init__(self,
                 unknown="<UNK>",
                 pad=None,
                 tokenizer=split,
                 preprocess=lower,
                 vocabulary=None,
                 min_frequency=1,
                 index_dtype=np.int32):
        if vocabulary is None:
            vocabulary = Vocab()
        if not preprocess:
            preprocess = raw
        self.index_dtype = index_dtype
        self._pad_id = vocabulary[pad] if pad is not None else -1
        self._wrapper = \
            _FrequentVocabWrapper(vocabulary, unknown, min_frequency) \
            if min_frequency > 1 else _VocabWrapper(vocabulary, unknown)
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._min_frequency = min_frequency

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def set_preprocess(self, func):
        self._preprocess = func

    def _fit_vocabulary(self, word):
        self._wrapper.fit(word)
        return self

    def _tranform_vocabulary(self, word):
        return self._wrapper.transform(word)

    def _fit_tranform_vocabulary(self, word):
        return self._wrapper.fit_transform(word)

    def fit(self, raw_document):
        for token in self._extract_tokens(raw_document):
            self._fit_vocabulary(token)
        return self

    def transform(self, raw_document, length=None):
        tokens = self._extract_tokens(raw_document)
        if length is not None:
            word_ids = np.full(length, self._pad_id, dtype=self.index_dtype)
            for i, token in enumerate(tokens):
                word_ids[i] = self._tranform_vocabulary(token)
        else:
            indices = [self._tranform_vocabulary(token) for token in tokens]
            word_ids = np.array(indices, self.index_dtype)
        return word_ids

    def fit_transform(self, raw_document, length=None):
        if self._min_frequency > 0:
            return self.fit(raw_document).transform(raw_document, length)
        else:
            ids = np.array([self._wrapper.fit_transform(token)
                            for token in self._extract_tokens(raw_document)],
                           self.index_dtype)
            return _np_pad(ids, length, self._pad_id,
                           self.index_dtype) if length else ids

    def _extract_tokens(self, raw_document):
        if isinstance(raw_document, str):
            tokens = self._tokenizer(raw_document)
        elif isinstance(raw_document, Iterable):
            tokens = raw_document
        else:
            raise ValueError(
                'raw_document must be an instance of str or Iterable')
        return (self._preprocess(token) for token in tokens)  # generator

    def pad(self, tokens, length):
        if isinstance(tokens, np.ndarray) or len(tokens) < 2:
            if length - len(tokens) < 0:
                raise ValueError(
                    "token length exceeds the specified length value")
            return _np_pad(tokens, length, self._pad_id, self.index_dtype)
        else:
            return np.vstack([
                _np_pad(array, length, self._pad_id, self.index_dtype)
                for array in tokens])

    def get_vocabulary_id(self, word):
        return self._tranform_vocabulary(word)

    @property
    def vocabulary(self):
        return self._wrapper.vocabulary.copy()

    @property
    def vocabulary_size(self):
        return self._wrapper.vocabulary.size

    @property
    def unknown_id(self):
        return self._wrapper.unknown_id

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def min_frequency(self):
        return self._min_frequency


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


_default_type = np.float32


def _get_int_type(dtype):
    if not isinstance(dtype, str):
        dtype = dtype.__name__
    return np.int64 if dtype == 'float64' else np.int32


def random_normal(shape, dtype=np.float32):
    return np.random.normal(0, 1, shape).astype(dtype, copy=False)


def random_uniform(shape, dtype=np.float32):
    return np.random.uniform(-1, 1, shape).astype(dtype, copy=False)


class EmbeddingPreprocessor(Preprocessor):

    def __init__(self,
                 embed_file=None,
                 embed_size=50,
                 unknown="<UNK>",
                 pad=None,
                 tokenizer=split,
                 initializer=random_normal,
                 preprocess=lower,
                 vocabulary=None,
                 min_frequency=1,
                 embed_dtype=np.float32):
        if embed_file is not None:
            vocab_file = None
            if isinstance(embed_file, (list, tuple)):
                embed_file, vocab_file = embed_file
            if vocabulary is not None:
                warnings.warn("vocabulary will be overwritten "
                              "by the dict of the embeddings")
            vocabulary, embeddings = \
                load_embeddings(embed_file, vocab_file, embed_dtype)
            self.use_pretrained = True
            embed_size = embeddings.shape[1]
        elif embed_size is not None:
            if embed_size <= 0 or type(embed_size) is not int:
                raise ValueError("embed_size must be a positive integer value")
            if vocabulary is None:
                vocabulary = Vocab()
            embeddings = initializer(
                (vocabulary.size, embed_size), embed_dtype)
            self.use_pretrained = False
        else:
            raise ValueError("embed_file os embed_size must be specified")
        self.embed_dtype = embed_dtype
        self._embeddings = embeddings
        self._embed_size = embed_size
        self._initializer = initializer
        self._deserialized = False

        super(EmbeddingPreprocessor, self).__init__(
            unknown, pad, tokenizer, preprocess,
            vocabulary, min_frequency, _get_int_type(embed_dtype))
        if not self.use_pretrained and self.pad_id >= 0:
            self.get_embeddings()
            self._embeddings[self._pad_id] = 0

    def reset_embeddings(self, embed_size):
        self._embeddings = self._initializer(
            (self.vocabulary_size, embed_size), self.embed_dtype)
        if self._pad_id >= 0:
            self._embeddings[self._pad_id] = 0
        self._embed_size = embed_size
        return self._embeddings

    def get_embeddings(self, normalize=False):
        # if self._deserialized:
        uninitialized_vocab_size = \
            self.vocabulary_size - self._embeddings.shape[0]
        if uninitialized_vocab_size > 0:
            new_vectors = self._initializer(
                (uninitialized_vocab_size, self._embed_size), self.embed_dtype)
            self._embeddings = np.r_[self._embeddings, new_vectors]
        if not normalize:
            embeddings = self._embeddings
        elif normalize == 'l2':
            l2 = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            l2[l2 == 0] = 1
            embeddings = self._embeddings / l2
        elif normalize == 'zscore':
            mean = np.mean(self._embeddings, axis=1, keepdims=True)
            std = np.std(self._embeddings, axis=1, keepdims=True)
            embeddings = (self._embeddings - mean) / std
        else:
            raise ValueError('unsupported normalization was specified: {}'
                             .format(normalize))
        return embeddings

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_embeddings']
        return state

    def __setstate__(self, state):
        state['_deserialized'] = True
        self.__dict__.update(state)
        self.reset_embeddings(self._embed_size)
