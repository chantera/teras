import hashlib
import json
import os
import pickle
import logging


class Cache(object):
    LOG_LEVEL = logging.DEBUG

    def __init__(self, key, dir='', ext='.pkl', prefix='', mkdir=False,
                 hash_length=16, serializer=None, deserializer=None,
                 logger=None):
        self._config = {
            'key': key,
            'dir': dir,
            'ext': ext,
            'prefix': prefix,
            'mkdir': mkdir,
            'hash_length': hash_length,
            'serializer': serializer,
            'deserializer': deserializer,
            'logger': logger,
        }
        dir = os.path.abspath(os.path.expanduser(dir))
        if os.path.isdir(dir):
            pass
        elif mkdir:
            os.makedirs(dir)
        else:
            raise FileNotFoundError(
                "cache dir was not found: `{}`".format(dir))
        self._identifier = self._encode_key(key)[:hash_length]
        self._file = os.path.join(dir, prefix + self._identifier + ext)
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

    def clone(self, key):
        return Cache(**dict(self._config, key=key))

    @property
    def id(self):
        return self._identifier

    @staticmethod
    def _encode_key(key):
        key_encoded = json.dumps(
            key, sort_keys=True, separators=(',', ':')).encode('utf-8')
        key_hash = hashlib.md5(key_encoded).hexdigest()
        return key_hash


def load(key, dir='', ext='.pkl', prefix='', hash_length=16,
         serializer=None, deserializer=None, logger=None):
    return Cache(key, dir, ext, prefix, False, hash_length,
                 serializer, deserializer, logger).load()


def load_or_create(key, factory, refresh=False,
                   dir='', ext='.pkl', prefix='', mkdir=False, hash_length=16,
                   serializer=None, deserializer=None, logger=None):
    return Cache(key, dir, ext, prefix, mkdir, hash_length,
                 serializer, deserializer, logger) \
        .load_or_create(factory, refresh)


def dump(obj, key, dir='', ext='.pkl', prefix='', mkdir=False, hash_length=16,
         serializer=None, deserializer=None, logger=None):
    Cache(key, dir, ext, prefix, mkdir, hash_length,
          serializer, deserializer, logger).dump(obj)
