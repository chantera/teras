import hashlib
import json
import os
import pickle
import logging


def _resolve_path(key, dir='', ext='', prefix='', hash_len=16):
    key_encoded = json.dumps(
        key, sort_keys=True, separators=(',', ':')).encode('utf-8')
    key_hash = hashlib.md5(key_encoded).hexdigest()
    path = os.path.join(dir, prefix + key_hash[:hash_len] + ext)
    return path


class Cache(object):
    LOG_LEVEL = logging.DEBUG

    def __init__(self, key, dir='', ext='.pkl', prefix='', mkdir=False,
                 serializer=None, deserializer=None, logger=None):
        dir = os.path.abspath(os.path.expanduser(dir))
        if os.path.isdir(dir):
            pass
        elif mkdir:
            os.makedirs(dir)
        else:
            raise FileNotFoundError(
                "cache dir was not found: `{}`".format(dir))
        self._file = _resolve_path(key, dir, ext, prefix)
        self._serializer = serializer \
            if serializer is not None else pickle
        self._deserializer = deserializer \
            if deserializer is not None else self._serializer
        self._logger = logger \
            if logger is not None else logging.getLogger(__name__)

    def load(self):
        self._logger.log(
            self.LOG_LEVEL, "loading cache from `{}`".format(self._file))
        with open(self._file, 'rb') as f:
            return self._serializer.load(f)

    def load_or_create(self, factory, refresh=False):
        if not refresh:
            try:
                return self.load()
            except FileNotFoundError:
                self._logger.log(
                    self.LOG_LEVEL, "missing cache - creating a new object")
        else:
            self._logger.log(
                self.LOG_LEVEL, "updating cache - creating a new object")
        obj = factory()
        self.dump(obj)
        return obj

    def dump(self, obj):
        self._logger.log(
            self.LOG_LEVEL, "dumping cache to `{}`".format(self._file))
        with open(self._file, 'wb') as f:
            self._serializer.dump(obj, f)


def load(key, dir='', ext='.pkl', prefix='',
         serializer=None, deserializer=None, logger=None):
    return Cache(key, dir, ext, prefix, False,
                 serializer, deserializer, logger).load()


def load_or_create(key, factory, refresh=False,
                   dir='', ext='.pkl', prefix='', mkdir=False,
                   serializer=None, deserializer=None, logger=None):
    return Cache(key, dir, ext, prefix, mkdir,
                 serializer, deserializer, logger) \
        .load_or_create(factory, refresh)


def dump(obj, key, dir='', ext='.pkl', prefix='', mkdir=False,
         serializer=None, deserializer=None, logger=None):
    Cache(key, dir, ext, prefix, mkdir, serializer, deserializer, logger) \
        .dump(obj)
