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
        self._processors[name] = text.Preprocessor(**kwargs)

    def get_processor(self, name):
        return self._processors[name]

    def get_embeddings(self, name):
        return self._processors[name].get_embeddings()

    @abstractmethod
    def map(self, item):
        raise NotImplementedError

    def filter(self, item):
        return True

    def load(self, file, train=False):
        self._train = train
        self._reader.set_file(file)
        samples = [self.map(item) for item in self._reader
                   if self.filter(item)]
        return Dataset(samples)
