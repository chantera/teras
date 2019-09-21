from collections.abc import Iterator
import os
import pathlib


class Reader(Iterator):

    def __init__(self, file=None):
        if file is not None:
            self.set_file(file)
        else:
            self.file = None
            self.reset()

    def set_file(self, file):
        if isinstance(file, pathlib.PurePath):
            file = str(file)
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
        raise NotImplementedError

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_iterator'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class LineReader(Reader):

    def _get_iterator(self):
        with open(self.file, mode='r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()


class CsvReader(Reader):

    def __init__(self, file=None, delimiter=','):
        super().__init__(file)
        self.delimiter = delimiter

    def _get_iterator(self):
        with open(self.file, mode='r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split(self.delimiter)


class ZipReader(Reader):

    def __init__(self, readers):
        self._readers = readers

    def set_file(self, file):
        if not isinstance(file, (tuple, list)):
            file = [file]
        return self.set_files(file)

    def set_files(self, files):
        if len(files) != len(self._readers):
            raise ValueError('files must be given as many as readers')
        self.reset()
        for reader, file in zip(self._readers, files):
            if file is not None:
                reader.set_file(file)
            else:
                reader.file = None

    def reset(self):
        for reader in self._readers:
            reader.reset()
        self._iterator = None

    def _get_iterator(self):
        def _yield_null():
            while True:
                yield None
        return zip(*[reader if reader.file is not None else _yield_null()
                     for reader in self._readers])


def _create_root(format='conll', extra_fields=None):
    if format == 'conll':
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
    elif format == 'conll09':
        root = {
            'id': 0,
            'form': "<ROOT>",
            'lemma': "<ROOT>",
            'plemma': "_",
            'pos': "ROOT",
            'ppos': "_",
            'feat': "_",
            'pfeat': "_",
            'head': 0,
            'phead': "_",
            'deprel': "root",
            'pdeprel': "_",
            'fillpred': "_",
            'pred': "_",
            'apreds': [],
        }
    else:
        raise ValueError("Format `` is not supported.".format(format))
    if extra_fields:
        _append_fields(root, extra_fields)
    return root


def _parse_conll(text, extra_fields=None):
    tokens = [_create_root('conll', extra_fields)]
    for line in [text] if isinstance(text, str) else text:
        line = line.strip()
        if not line:
            if len(tokens) > 1:
                yield tokens
                tokens = [_create_root('conll', extra_fields)]
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
            if extra_fields:
                _append_fields(token, extra_fields, cols)
            tokens.append(token)
    if len(tokens) > 1:
        yield tokens


def _append_fields(token, fields, cols=None):
    for name, field in fields.items():
        if isinstance(field, dict):
            if cols is None:
                if token['id'] == 0 and 'root' in field:
                    value = field['root']
                elif 'default' in field:
                    value = field['default']
                else:
                    raise IndexError('cannot extract field')
            else:
                index = field['index']
                assert isinstance(index, int)
                if len(cols) <= index and 'default' in field:
                    value = field['default']
                else:
                    # This raises IndexError if len(cols) <= field
                    value = cols[index]
        else:
            assert isinstance(field, int)
            if cols is None and token['id'] == 0:
                value = None
            else:
                # This raises raise IndexError if len(cols) <= field
                value = cols[field]
        token[name] = value


def _parse_conll09(text, extra_fields=None):
    if extra_fields is not None:
        raise NotImplementedError
    tokens = [_create_root(format='conll09')]
    for line in [text] if isinstance(text, str) else text:
        line = line.strip()
        if not line:
            if len(tokens) > 1:
                yield tokens
                tokens = [_create_root(format='conll09')]
        elif line.startswith('#'):
            continue
        else:
            cols = line.split("\t")
            token = {
                'id': int(cols[0]),
                'form': cols[1],
                'lemma': cols[2],
                'plemma': cols[3],
                'pos': cols[4],
                'ppos': cols[5],
                'feat': cols[6],
                'pfeat': cols[7],
                'head': int(cols[8]),
                'phead': cols[9],
                'deprel': cols[10],
                'pdeprel': cols[11],
                'fillpred': cols[12],
                'pred': cols[13],
                'apreds': cols[14:],
            }
            tokens.append(token)
    if len(tokens) > 1:
        yield tokens


class ConllReader(Reader):

    def __init__(self, file=None, format='conll', extra_fields=None):
        super().__init__(file)
        self.format = format
        self.extra_fields = extra_fields

    def _get_iterator(self):
        if self.format == 'conll':
            parse_func = _parse_conll
        elif self.format == 'conll09':
            parse_func = _parse_conll09
        else:
            raise ValueError("Format `` is not supported.".format(format))
        with open(self.file, mode='r', encoding='utf-8') as f:
            yield from parse_func(f, self.extra_fields)


def read_conll(file, format='conll', extra_fields=None):
    if isinstance(file, pathlib.PurePath):
        file = str(file)
    with open(file, mode='r', encoding='utf-8') as f:
        return parse_conll(f, format, extra_fields)


def parse_conll(text, format='conll', extra_fields=None):
    if format == 'conll':
        return list(_parse_conll(text, extra_fields))
    elif format == 'conll09':
        return list(_parse_conll09(text, extra_fields))
    else:
        raise ValueError("Format `` is not supported.".format(format))


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
        with open(self.file, mode='r', encoding='utf-8') as f:
            yield from _parse_tree(f, self.left_bracket, self.right_bracket)


def read_tree(file, left_bracket='(', right_bracket=')'):
    if isinstance(file, pathlib.PurePath):
        file = str(file)
    with open(file, mode='r', encoding='utf-8') as f:
        return list(_parse_tree(f, left_bracket, right_bracket))


def parse_tree(text, left_bracket='(', right_bracket=')'):
    return list(_parse_tree(text, left_bracket, right_bracket))


class ContextualizedEmbeddingsReader(Reader):

    def _get_iterator(self):
        with ContextualizedEmbeddingsFile.open(self.file) as f:
            for value in f:
                yield value


class ContextualizedEmbeddingsFile(object):

    def __init__(self, file):
        handle = None
        try:
            import h5py
            handle = h5py.File(file, 'r')
        except Exception as e:
            raise e
        self._file = file
        self._handle = handle
        self._sentence_id = -1

    @classmethod
    def open(cls, file):
        return cls(file)

    def close(self):
        if not self.closed:
            self._handle.close()
            self._handle = None

    @property
    def closed(self):
        return self._handle is None

    def _check_closed(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        self._check_closed()
        return self

    def __next__(self):
        self._check_closed()
        self._sentence_id += 1
        key = str(self._sentence_id)
        if key not in self._handle:
            raise StopIteration
        value = self._handle[key][...]
        return value
