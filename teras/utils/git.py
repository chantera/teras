from functools import lru_cache
import os
from subprocess import check_output, CalledProcessError, STDOUT


def _check(path):
    if type(path) is not str:
        raise ValueError("path must be specified as str (actual: {})"
                         .format(type(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no such file or directory: {}".format(path))
    return True


def _split(path):
    if os.path.isdir(path):
        _dir, _file = path.rstrip(os.path.sep), None
    else:
        _dir, _file = os.path.split(path)
    return _dir, _file


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
    if path is not None:
        _check(path)
        kwargs['cwd'] = _split(path)[0]

    return _exec(command, suppress_error, **kwargs)


@lru_cache(maxsize=1)
def relpath(path, suppress_error=True):
    command = ['git', 'rev-parse', '--show-prefix']
    kwargs = {}
    if path is not None:
        _check(path)
        kwargs['cwd'], _file = _split(path)

    _relpath = _exec(command, suppress_error, **kwargs)
    if _relpath is not None and _file is not None:
        _relpath = os.path.join(_relpath, _file)
    return _relpath
