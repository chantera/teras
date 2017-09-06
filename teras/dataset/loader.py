from abc import ABC, abstractmethod

from teras.dataset import Dataset
from teras.preprocessing import text


class Loader(ABC):

    def load(self, file):
        raise NotImplementedError


class CorpusLoader(Loader):

    def __init__(self, reader):
        self._reader = reader
        self._processors = {}
        self._train = False

    def add_processor(self, name, **kwargs):
        for cls in (text.EmbeddingPreprocessor, text.Preprocessor):
            try:
                self._processors[name] = cls(**kwargs)
                break
            except TypeError:
                continue
        else:
            raise TypeError('Preprocessor not found for the specified args: {}'
                            .format(kwargs))

    def get_processor(self, name):
        return self._processors[name]

    def get_embeddings(self, name, normalize=False):
        if not isinstance(self._processors[name], text.EmbeddingPreprocessor):
            raise TypeError('preprocessor[name={}] '
                            'is not an EmbeddingPreprocessor'.format(name))
        return self._processors[name].get_embeddings(normalize)

    @abstractmethod
    def map(self, item):
        raise NotImplementedError

    def filter(self, item):
        return True

    def load(self, file, train=False, size=None):
        self._train = train
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

        return Dataset(samples)
