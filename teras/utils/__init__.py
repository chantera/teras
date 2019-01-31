import dill
import os

from teras.utils import abc  # NOQA
from teras.utils import builtin  # NOQA
from teras.utils import collections  # NOQA
from teras.utils import decorators  # NOQA
from teras.utils import git  # NOQA
from teras.utils import progressbar  # NOQA


# __all__ = ['builtin', 'dump', 'git', 'load', 'load_context', 'progressbar']


def dump(obj, file, **kwargs):
    return dill.dump(obj, file, **kwargs)


def load(file):
    return dill.load(file)


def load_context(model_file):
    _dir, _file = os.path.split(model_file)
    context_file = os.path.basename(_file).split('.')[0] + '.context'
    context_file = os.path.join(_dir, context_file)
    with open(context_file, 'rb') as f:
        context = collections.ImmutableMap(load(f))
    return context
