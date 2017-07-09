import dill

from teras.utils import builtin
from teras.utils import git
from teras.utils import progressbar

__all__ = ['builtin', 'git', 'progressbar']


def dump(obj, file, **kwargs):
    return dill.dump(obj, file, **kwargs)


def load(file):
    return dill.load(file)
