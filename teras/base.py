from abc import ABCMeta, abstractmethod
from collections.abc import Callable


class Singleton(metaclass=ABCMeta):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = \
                super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


class Observable(object):

    def __init__(self):
        self._hooks = {}

    def add_hook(self, event, hook):
        if event not in self._hooks:
            self._hooks[event] = [hook]
        elif hook not in self._hooks[event]:
            self._hooks[event].append(hook)

    def notify(self, event, data=None):
        if event in self._hooks:
            for hook in self._hooks[event]:
                hook(data)

    def remove_hook(self, event, hook):
        if event in self._hooks and hook in self._hooks[event]:
            self._hooks[event].remove(hook)


class Context(dict, Callable):

    def __call__(self, key, default=None):
        if default is not None and key not in self:
            return default
        return self[key]


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
