from collections import Iterable, UserDict
import copy
import re

import numpy as np


class Vocab(UserDict):

    def __init__(self):
        super(Vocab, self).__init__()
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
        return self.__len__()

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
                 index_dtype=np.int32):
        if not hasattr(self, '_vocabulary'):
            self._vocabulary = Vocab()
        if not preprocess:
            preprocess = raw
        self.index_dtype = index_dtype
        self._pad_id = -1 if pad is None \
            else self._add_vocabulary(preprocess(pad))
        self._unknown_id = self._add_vocabulary(preprocess(unknown))
        self._tokenizer = tokenizer
        self._preprocess = preprocess

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def set_preprocess(self, func):
        self._preprocess = func

    def _add_vocabulary(self, word):
        return self._vocabulary[word]

    def fit(self, raw_documents, as_one=False):
        if isinstance(raw_documents, str) or as_one:
            self._fit_each(raw_documents)
        else:
            for document in raw_documents:
                self._fit_each(document)
        return self

    def _fit_each(self, raw_document):
        for token in self._extract_tokens(raw_document):
            self._add_vocabulary(token)

    def transform(self, raw_documents, length=None, as_one=False):
        if isinstance(raw_documents, str) or as_one:
            return self._transform_each(raw_documents, length)
        else:
            samples = [self._transform_each(document, length)
                       for document in raw_documents]
            return np.array(samples, self.index_dtype) if length else samples

    def _transform_each(self, raw_document, length=None):
        tokens = self._extract_tokens(raw_document)
        if length is not None:
            word_ids = np.full(length, self._pad_id, dtype=self.index_dtype)
            for i, token in enumerate(tokens):
                word_ids[i] = self.get_vocabulary_id(token)
        else:
            indices = [self.get_vocabulary_id(token) for token in tokens]
            word_ids = np.array(indices, self.index_dtype)
        return word_ids

    def fit_transform(self, raw_documents, length=None, as_one=False):
        if isinstance(raw_documents, str) or as_one:
            ids = np.array([self._add_vocabulary(token)
                            for token in self._extract_tokens(raw_documents)],
                           self.index_dtype)
            return _np_pad(ids, length, self._pad_id,
                           self.index_dtype) if length else ids

        elif length:
            return np.vstack([
                _np_pad(list(map(self._add_vocabulary,
                                 self._extract_tokens(raw_document))),
                        length,
                        self._pad_id,
                        self.index_dtype)
                for raw_document in raw_documents])
        else:
            return [
                np.array(list(map(self._add_vocabulary,
                                  self._extract_tokens(raw_document))),
                         self.index_dtype)
                for raw_document in raw_documents]

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
        return self._vocabulary.get(word, self._unknown_id)

    @property
    def vocabulary(self):
        return self._vocabulary.copy()

    @property
    def vocabulary_size(self):
        return self._vocabulary.size

    @property
    def unknown_id(self):
        return self._unknown_id

    @property
    def pad_id(self):
        return self._pad_id


def load_embeddings(embed_file, vocab_file=None, dtype=np.float32):
    vocabulary = Vocab()
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


class Initializer(object):

    def __call__(self, shape, dtype=np.float32):
        raise NotImplementedError


class Normal(Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape, dtype=np.float32):
        return np.random.normal(0, self.scale, shape) \
            .astype(dtype, copy=False)


class Uniform(Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape, dtype=np.float32):
        return np.random.uniform(-1 * self.scale, 1 * self.scale, shape) \
            .astype(dtype, copy=False)


def standard_normal(shape, dtype=np.float32):
    return np.random.normal(0, 1, shape).astype(dtype, copy=False)


class EmbeddingPreprocessor(Preprocessor):

    def __init__(self,
                 embed_file=None,
                 embed_size=50,
                 unknown="<UNK>",
                 pad=None,
                 tokenizer=split,
                 initializer=standard_normal,
                 preprocess=lower,
                 embed_dtype=np.float32):
        self.embed_dtype = embed_dtype
        self._init_embeddings(embed_file, embed_size)
        super(EmbeddingPreprocessor, self).__init__(
            unknown, pad, tokenizer, preprocess, _get_int_type(embed_dtype))
        self._initializer = initializer
        if not self.use_pretrained and self._pad_id >= 0:
            self.get_embeddings()[self._pad_id] = 0

    def _init_embeddings(self, embed_file, embed_size):
        if embed_file is not None:
            vocab_file = None
            if isinstance(embed_file, (list, tuple)):
                embed_file, vocab_file = embed_file
            vocabulary, embeddings = \
                load_embeddings(embed_file, vocab_file, self.embed_dtype)
            self.use_pretrained = True
            embed_size = embeddings.shape[1]
        elif embed_size is not None:
            if embed_size <= 0 or type(embed_size) is not int:
                raise ValueError("embed_size must be a positive integer value")
            vocabulary, embeddings = \
                Vocab(), np.zeros((0, embed_size), self.embed_dtype)
            self.use_pretrained = False
        else:
            raise ValueError("embed_file os embed_size must be specified")

        self._vocabulary = vocabulary
        self._embeddings = embeddings
        self._new_words = []
        self._embed_size = embed_size

    def reset_embeddings(self, embed_size):
        self._embeddings = self._initializer(
            (len(self._vocabulary), embed_size), self.embed_dtype)
        if self._pad_id >= 0:
            self._embeddings[self._pad_id] = 0
        self._embed_size = embed_size
        return self._embeddings

    def _add_vocabulary(self, word):
        if word not in self._vocabulary:
            index = self._vocabulary[word]
            self._new_words.append(index)
        else:
            index = self._vocabulary[word]
        return index

    def get_embeddings(self):
        uninitialized_vocab_size = len(self._new_words)
        if uninitialized_vocab_size > 0:
            new_vectors = self._initializer(
                (uninitialized_vocab_size, self._embed_size), self.embed_dtype)
            self._embeddings = np.r_[self._embeddings, new_vectors]
            self._new_words = []
        return self._embeddings

    @property
    def embeddings(self):
        return self.get_embeddings()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_new_words'] = []
        del state['_embeddings']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reset_embeddings()
        self._embeddings = np.zeros(
            (self._vocabulary.size, self._embed_size), dtype=self.embed_dtype)
