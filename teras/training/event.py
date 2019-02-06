from collections.abc import Callable
from enum import Enum
from types import MethodType


class Event(Enum):

    def __str__(self):
        return self.value


class Listener(Callable):

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


class Dispatcher(object):
    EventClass = Event

    def __init__(self):
        self._hooks = {}
        self._priorities = {}
        self._listeners = {}

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

    def add_listener(self, listener, priority=100, update=False):
        self.check_listener(listener)
        if update:
            self.detach_listener(listener)
        if listener.name not in self._listeners:
            for event in self.EventClass:
                if listener.has_handler(event):
                    self.add_hook(event, listener.get_handler(event), priority)
            self._listeners[listener.name] = listener

    def remove_listener(self, listener):
        if isinstance(listener, str):
            listener = self.get_listener(listener)
        self.check_listener(listener)
        if listener.name in self._listeners:
            for event in self.EventClass:
                if listener.has_handler(event):
                    self.remove_hook(event, listener.get_handler(event))
            del self._listeners[listener.name]

    def has_listener(self, name):
        return name in self._listeners

    def get_listener(self, name):
        return self._listeners[name]

    @staticmethod
    def check_listener(listener):
        if not isinstance(listener, Listener):
            raise ValueError("listener is not a Listener object: {}"
                             .format(listener))


class TrainEvent(Event):
    TRAIN_BEGIN = 'train_begin'
    TRAIN_END = 'train_end'
    EPOCH_BEGIN = 'epoch_begin'
    EPOCH_END = 'epoch_end'
    EPOCH_TRAIN_BEGIN = 'epoch_train_begin'
    EPOCH_TRAIN_END = 'epoch_train_end'
    EPOCH_VALIDATE_BEGIN = 'epoch_validate_begin'
    EPOCH_VALIDATE_END = 'epoch_validate_end'
    BATCH_BEGIN = 'batch_begin'
    BATCH_END = 'batch_end'
