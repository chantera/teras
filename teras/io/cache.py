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

    def __init__(self, key, dir='', ext='.pkl', prefix='',
                 serializer=None, deserializer=None, logger=None):
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

    def load_or_create(self, factory):
        try:
            self.load()
        except FileNotFoundError:
            self._logger.log(
                self.LOG_LEVEL, "missing cache - creating a new object")
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
    return Cache(key, dir, ext, prefix, serializer, deserializer, logger) \
        .load()


def load_or_create(key, factory, dir='', ext='.pkl', prefix='',
                   serializer=None, deserializer=None, logger=None):
    return Cache(key, dir, ext, prefix, serializer, deserializer, logger) \
        .load_or_create(factory)


def dump(obj, key, dir='', ext='.pkl', prefix='',
         serializer=None, deserializer=None, logger=None):
    Cache(key, dir, ext, prefix, serializer, deserializer, logger).dump(obj)
