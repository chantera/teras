from functools import lru_cache
import os
from subprocess import check_output, CalledProcessError, STDOUT


def _check(path):
    if type(path) is not str:
        raise ValueError(
            "path must be specified as str (actual: {})".format(type(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no such file or directory: {}".format(path))
    return True


def _split(path):
    if os.path.isdir(path):
        dir, file = path.rstrip(os.path.sep), None
    else:
        dir, file = os.path.split(path)
    return dir, file


def _set_cwd(d, path=None, check=True):
    if path is not None:
        if check:
            _check(path)
        dir = _split(path)[0]
        if dir != '':
            d['cwd'] = dir


def _exec(command, suppress_error=True, **kwargs):
    try:
        output = check_output(command, stderr=STDOUT, shell=False, **kwargs)
    except CalledProcessError as exc:
        if not suppress_error:
            raise OSError(exc.output.decode('utf-8'))
        return None
    return output.decode('utf-8').strip()


@lru_cache(maxsize=1)
def root(path=None, suppress_error=True):
    command = ['git', 'rev-parse', '--show-toplevel']
    kwargs = {}
    _set_cwd(kwargs, path)
    return _exec(command, suppress_error, **kwargs)


@lru_cache(maxsize=1)
def hash(path=None, short=False, suppress_error=True):
    command = ['git', 'rev-parse',
               ('--short' if short else '--verify'), 'HEAD']
    kwargs = {}
    _set_cwd(kwargs, path)
    return _exec(command, suppress_error, **kwargs)


@lru_cache(maxsize=1)
def relpath(path=None, suppress_error=True):
    command = ['git', 'rev-parse', '--show-prefix']
    kwargs = {}
    _set_cwd(kwargs, path)
    file = _split(path)[1] if path else None
    relpath = _exec(command, suppress_error, **kwargs)
    if relpath is not None and file is not None:
        relpath = os.path.join(relpath, file)
    return relpath
