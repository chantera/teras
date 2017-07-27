from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Mapping


class Singleton(metaclass=ABCMeta):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = \
                super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


class ImmutableDict(dict):

    def __setitem__(self, key, value):
        raise TypeError("'{}' object does not support item assignment"
                        .format(type(self).__name__))

    def __delitem__(self, key):
        raise TypeError("'{}' object does not support item deletion"
                        .format(type(self).__name__))

    def clear(self):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, 'clear'))

    def update(self, *args, **kwargs):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, 'update'))

    def setdefault(self, key, default=None):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, 'setdefault'))

    def pop(self, key, default=None):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, 'pop'))

    def popitem(self):
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(type(self).__name__, 'popitem'))

    def __hash__(self):
        return hash(tuple(sorted(self.iteritems())))


class Context(Mapping, Callable):

    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'Context' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        if args:
            d = dict(args[0])
            if len(kwargs):
                d.update(kwargs)
        elif len(kwargs):
            d = kwargs
        else:
            d = {}
        self.data = ImmutableDict(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        raise KeyError(key)

    def __iter__(self):
        return iter(self.data)

    def __call__(self, key, default=None):
        if default is not None and key not in self:
            return default
        return self[key]

    def __getattr__(self, attr):
        if attr in self.data:
            return self.data[attr]
        if not hasattr(self.data, attr):
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(type(self).__name__, attr))
        return getattr(self.data, attr)

    def __getstate__(self):
        d = dict()
        d.update(self.data)
        return d

    def __setstate__(self, state):
        self.data = ImmutableDict(state)

    def __repr__(self):
        return self.data.__repr__()


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
