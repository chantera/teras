from abc import ABC, abstractmethod

from teras.dataset.dataset import Dataset, BucketDataset
from teras.preprocessing import text


class Loader(ABC):

    def load(self, file):
        raise NotImplementedError


class CorpusLoader(Loader):

    def __init__(self, reader):
        self._reader = reader
        self._processors = {}
        self.train = False

    def add_processor(self, name, **kwargs):
        self._processors[name] = text.Preprocessor(**kwargs)

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
        if size is None:
            samples = [self.map(item) for item in self._reader
                       if self.filter(item)]
        else:
            samples = []
            for item in self._reader:
                if len(samples) >= size:
                    break
                if not self.filter(item):
                    continue
                samples.append(self.map(item))

        if bucketing:
            return BucketDataset(samples, key=0, equalize_by_key=False)
        else:
            return Dataset(samples)
