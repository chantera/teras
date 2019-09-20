import copy
from collections.abc import MutableMapping, Mapping


class ImmutableDict(dict):

    def __setitem__(self, key, value):
        raise TypeError("'{}' object does not support item assignment"
                        .format(type(self).__name__))

    def __delitem__(self, key):
        raise TypeError("'{}' object does not support item deletion"
                        .format(type(self).__name__))

    def clear(self):
        raise TypeError("'{}' object does not support item deletion"
                        .format(type(self).__name__))

    def update(self, *args, **kwargs):
        raise TypeError("'{}' object does not support item assignment"
                        .format(type(self).__name__))

    def setdefault(self, key, default=None):
        raise TypeError("'{}' object does not support item assignment"
                        .format(type(self).__name__))

    def pop(self, key, default=None):
        raise TypeError("'{}' object does not support item deletion"
                        .format(type(self).__name__))

    def popitem(self):
        raise TypeError("'{}' object does not support item deletion"
                        .format(type(self).__name__))

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls((key, value) for key in iterable)


class MutableMap(MutableMapping):

    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'MutableMap' object "
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
        self.data = d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def clear(self):
        self.data.clear()

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self.data.setdefault(key, default)

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def popitem(self):
        return self.data.popitem()

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __getstate__(self):
        d = dict()
        d.update(self.data)
        return d

    def __setstate__(self, state):
        self.data = dict(state)

    def __repr__(self):
        return self.data.__repr__()

    def copy(self):
        if self.__class__ is MutableMap:
            return MutableMap(self.data.copy())
        data = self.data
        try:
            self.data = {}
            c = copy.copy(self)
        finally:
            self.data = data
        c.update(self)
        return c

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls((key, value) for key in iterable)


class ImmutableMap(Mapping):

    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'ImmutableMap' object "
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

    def __contains__(self, key):
        return key in self.data

    def __getstate__(self):
        d = dict()
        d.update(self.data)
        return d

    def __setstate__(self, state):
        self.data = ImmutableDict(state)

    def __repr__(self):
        return self.data.__repr__()

    def copy(self):
        if self.__class__ is ImmutableMap:
            return ImmutableMap(self.data.copy())
        data = self.data
        try:
            self.data = {}
            c = copy.copy(self)
        finally:
            self.data = data
        dict.update(c.data, self)
        return c

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls((key, value) for key in iterable)
