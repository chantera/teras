import sys

from progressbar import ProgressBar as PBar


class ProgressBar(object):

    def __init__(self, fd=sys.stderr):
        self._fd = fd
        self._pbar = None

    def start(self, max_value):
        self._pbar = \
            PBar(min_value=0, max_value=max_value, fd=self._fd).start()

    def update(self, count):
        self._pbar.update(count)

    def finish(self):
        self._pbar.finish()
