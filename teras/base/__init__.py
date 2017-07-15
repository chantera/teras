from abc import ABCMeta, abstractmethod
from collections.abc import Callable


class Singleton(metaclass=ABCMeta):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = \
                super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


class Iterable(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class Iterator(Iterable):

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError


class Context(dict, Callable):

    def __call__(self, key, default=None):
        if default is not None and key not in self:
            return default
        return self[key]

    def __getattr__(self, attr):
        return self[attr]


class Runner(Callable):

    def __init__(self, context):
        self._context = context
        self._initialize()

    def _initialize(self):
        pass

    def setup(self):
        pass

    def teardown(self):
        pass

    def __call__(self):
        with self as s:
            s.process()

    @abstractmethod
    def process(self):
        raise NotImplementedError()

    def __enter__(self):
        self.setup()

    def __exit__(self):
        self.teardown()
