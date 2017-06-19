from collections.abc import Callable
from enum import Enum
from types import MethodType


class Event(Enum):

    def __str__(self):
        return self.value


class EventHandler(Callable):

    def __call__(self, data=None):
        pass


class Callback(Callable):

    def __init__(self, name, **kwargs):
        self.name = name
        for k, v in kwargs.items():
            if callable(v):
                self.implement(k, v)

    def implement(self, event, func):
        method_name = self.resolve_handler_name(event)
        if isinstance(func, MethodType):
            if func.__self__ is not self:
                raise ValueError("func is alreary bound to another object: "
                                 "{}".format(func))
            setattr(self, method_name, func)
        elif callable(func):
            setattr(self, method_name, MethodType(func, self))
        else:
            raise ValueError("func is not callable: {}".format(func))
        return True

    def has_handler(self, event):
        method_name = self.resolve_handler_name(event)
        return hasattr(self, method_name)

    def get_handler(self, event):
        method_name = self.resolve_handler_name(event)
        return getattr(self, method_name)

    def __call__(self, event, data=None):
        return self.get_handler(event)(data)

    @staticmethod
    def resolve_handler_name(event):
        event = str(event)
        if not event.startswith('on_'):
            return 'on_' + event
        return event


class Observable(object):

    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        if not callable(observer):
            raise ValueError("observer is not callable: {}".format(observer))
        self._observers.append(observer)

    def update(self):
        for observer in self._observers:
            observer(self)


class EventSender(object):
    EventClass = Event

    def __init__(self):
        self._hooks = {}
        self._priorities = {}
        self._callbacks = {}

    def add_hook(self, event, hook, priority=100):
        self._priorities[(event, hook)] = priority
        if event not in self._hooks:
            self._hooks[event] = [hook]
        elif hook not in self._hooks[event]:
            self._hooks[event].append(hook)
            self._hooks[event].sort(
                key=lambda x: self._priorities[(event, x)], reverse=True)

    def notify(self, event, data=None):
        if event in self._hooks:
            for hook in self._hooks[event]:
                hook(data)

    def remove_hook(self, event, hook):
        if event in self._hooks and hook in self._hooks[event]:
            self._hooks[event].remove(hook)
            del self._hooks[(event, hook)]

    def attach_callback(self, callback, priority=100, update=False):
        self.check_callback(callback)
        if update:
            self.detach_callback(callback)
        if callback.name not in self._callbacks:
            for event in self.EventClass:
                if callback.has_handler(event):
                    self.add_hook(event, callback.get_handler(event), priority)
            self._callbacks[callback.name] = callback

    def detach_callback(self, callback):
        self.check_callback(callback)
        if callback.name in self._callbacks:
            for event in self.EventClass:
                if callback.has_handler(event):
                    self.remove_hook(event, callback.get_handler(event))
            del self._callbacks[callback.name]

    @staticmethod
    def check_callback(callback):
        if not isinstance(callback, Callback):
            raise ValueError("callback is not a Callback object: {}"
                             .format(callback))


class EventDispatcher(EventSender):
    pass
