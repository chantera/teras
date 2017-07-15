import os

from teras.base import Iterator


class Reader(Iterator):

    def __init__(self, file=None):
        self.set_file(file)

    def set_file(self, file):
        file = os.path.expanduser(file)
        if not os.path.exists(file):
            raise FileNotFoundError("file was not found: '{}'"
                                    .format(file))
        self.file = file
        self.reset()

    def __iter__(self):
        self._iterator = self._get_iterator()
        return self

    def __next__(self):
        return self._iterator.__next__()

    def read(self, file=None):
        if file is not None:
            self.set_file(file)
        items = [item for item in self]
        return items

    def read_next(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
        return self._iterator.__next__()

    def reset(self):
        self._iterator = None

    def _get_iterator(self):
        with open(self.file, mode='r') as f:
            for line in f:
                yield line.strip()


class ConllReader(Reader):

    def create_root(self):
        root = {
            'id': 0,
            'form': "<ROOT>",
            'lemma': "<ROOT>",
            'cpostag': "ROOT",
            'postag':  "ROOT",
            'feats':   "_",
            'head': 0,
            'deprel':  "root",
            'phead':   "_",
            'pdeprel': "_",
        }
        return root

    def _get_iterator(self):
        with open(self.file, mode='r') as f:
            tokens = [self.create_root()]
            for line in f:
                line = line.strip()
                if not line:
                    if len(tokens) > 1:
                        yield tokens
                        tokens = [self.create_root()]
                elif line.startswith('#'):
                    continue
                else:
                    cols = line.split("\t")
                    token = {
                        'id': int(cols[0]),
                        'form': cols[1],
                        'lemma': cols[2],
                        'cpostag': cols[3],
                        'postag': cols[4],
                        'feats': cols[5],
                        'head': int(cols[6]),
                        'deprel': cols[7],
                        'phead': cols[8],
                        'pdeprel': cols[9],
                    }
                    tokens.append(token)
            if len(tokens) > 1:
                yield tokens
