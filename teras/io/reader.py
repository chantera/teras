from collections.abc import Iterator
import os


class Reader(Iterator):

    def __init__(self, file=None):
        if file is not None:
            self.set_file(file)
        else:
            self.file = None
            self.reset()

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
        try:
            return self._iterator.__next__()
        except Exception as e:
            self.reset()
            raise e

    def read(self, file=None):
        if file is not None:
            self.set_file(file)
        items = [item for item in self]
        return items

    def read_next(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
        return self.__next__()

    def reset(self):
        self._iterator = None

    def _get_iterator(self):
        with open(self.file, mode='r') as f:
            for line in f:
                yield line.strip()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_iterator'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def _create_root():
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


def _parse_conll(text):
    tokens = [_create_root()]
    for line in [text] if isinstance(text, str) else text:
        line = line.strip()
        if not line:
            if len(tokens) > 1:
                yield tokens
                tokens = [_create_root()]
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


class ConllReader(Reader):

    def _get_iterator(self):
        with open(self.file, mode='r') as f:
            yield from _parse_conll(f)


def read_conll(file):
    with open(file, mode='r') as f:
        return list(_parse_conll(f))


def parse_conll(text):
    return list(_parse_conll(text))


def _parse_tree(text, left_bracket='(', right_bracket=')'):
    stack = []
    _buffer = []
    for line in [text] if isinstance(text, str) else text:
        line = line.lstrip()
        if not line:
            continue
        for char in line:
            if char == left_bracket:
                stack.append([])
            elif char == ' ' or char == '\n':
                if _buffer:
                    stack[-1].append(''.join(_buffer))
                    _buffer = []
            elif char == right_bracket:
                if _buffer:
                    stack[-1].append(''.join(_buffer))
                    _buffer = []
                if len(stack) > 1:
                    stack[-2].append(stack.pop())
                else:
                    yield stack.pop()
            else:
                _buffer.append(char)


class TreeReader(Reader):

    def __init__(self, file=None, left_bracket='(', right_bracket=')'):
        super(TreeReader, self).__init__(file)
        self.right_bracket = right_bracket
        self.left_bracket = left_bracket

    def _get_iterator(self):
        with open(self.file, mode='r') as f:
            yield from _parse_tree(f, self.left_bracket, self.right_bracket)


def read_tree(file, left_bracket='(', right_bracket=')'):
    with open(file, mode='r') as f:
        return list(_parse_tree(f, left_bracket, right_bracket))


def parse_tree(text, left_bracket='(', right_bracket=')'):
    return list(_parse_tree(text, left_bracket, right_bracket))
