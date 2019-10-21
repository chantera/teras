from abc import ABC, abstractmethod

from teras.dataset.dataset import Dataset, BucketDataset
from teras.io.cache import Cache
from teras.preprocessing import text


class Loader(ABC):

    def load(self, file):
        raise NotImplementedError


class TextLoader(Loader):

    def __init__(self, reader):
        self._reader = reader
        self._processors = {}
        self.train = False
        self._context = None

    def add_processor(self, name, *args, **kwargs):
        self._processors[name] = text.Preprocessor(*args, **kwargs)

    def get_processor(self, name):
        return self._processors[name]

    def get_embeddings(self, name, normalize=False):
        if isinstance(self._processors[name].vocab, text.EmbeddingVocab):
            return self._processors[name].vocab.get_embeddings(normalize)
        return None

    @abstractmethod
    def map(self, item):
        raise NotImplementedError

    def map_attr(self, attr, value, update=True):
        return self._processors[attr].fit_transform(value, fit=update)

    def filter(self, item):
        return True

    def load(self, file, train=False, size=None, bucketing=False):
        self.train = train
        self._reader.set_file(file)
        self._context = {
            'train': train,
            'item_index': -1,
            'num_samples': 0,
        }

        def _next_sample(reader):
            context = self._context
            for item in reader:
                context['item_index'] += 1
                if self.filter(item):
                    sample = self.map(item)
                    context['num_samples'] += 1
                    yield sample

        if size is None:
            samples = list(_next_sample(self._reader))
        else:
            samples = []
            for sample in _next_sample(self._reader):
                samples.append(sample)
                if len(samples) >= size:
                    break

        self._context = None
        if bucketing:
            return BucketDataset(samples, key=0, equalize_by_key=True)
        else:
            return Dataset(samples)


class CachedTextLoader(TextLoader):
    DEFAULT_CACHE_OPTIONS = {
        'key': None,
        'dir': None,
        'ext': '.pkl',
        'prefix': '',
        'mkdir': False,
        'hash_length': 16,
        'serializer': None,
        'deserializer': None,
        'logger': None,
    }
    _cache_io = None

    @classmethod
    def build(cls, enable_cache=True, cache_options=None, extra_ids=None,
              refresh_cache=False, **kwargs):
        cache_io = None
        if enable_cache:
            if cache_options is None:
                cache_options = dict()
            cache_options = {**cls.DEFAULT_CACHE_OPTIONS, **cache_options}
            if cache_options['dir'] is None:
                raise FileNotFoundError("cache dir was must be specified")
            elif cache_options['key'] is None:
                cache_options['key'] = dict(kwargs, extra_ids=extra_ids)
            cache_io = Cache(**cache_options)

        def _instantiate():
            instance = cls(**kwargs)
            instance._cache_io = cache_io
            return instance

        if cache_io is None:
            return _instantiate()
        else:
            instance = cache_io.load_or_create(_instantiate, refresh_cache)
            if instance._cache_io is None:
                instance._cache_io = cache_io
            return instance

    def update_cache(self):
        if self._cache_io is None:
            raise RuntimeError('caching is not enabled')
        self._cache_io.dump(self)

    def load(self, file, train=False, size=None, bucketing=False,
             extra_ids=None, refresh_cache=False, disable_cache=False):
        def _load():
            return super(CachedTextLoader, self) \
                .load(file, train, size, bucketing)

        if self._cache_io is None or disable_cache:
            return _load()
        else:
            hash_key = (self._cache_io.id,
                        file, train, size, bucketing, extra_ids)
            cache_io = self._cache_io.clone(hash_key)
            return cache_io.load_or_create(_load, refresh_cache)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache_io'] = None
        return state
