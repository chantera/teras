class State:

    def __init__(self, sentence):
        self._sentence = sentence
        self._num_tokens = len(sentence)
        self._stack = [0]
        self._buffer = 1
        self._heads = [0] * self._num_tokens
        self._labels = [0] * self._num_tokens
        self._history = []

    def advance(self):
        self._buffer += 1

    def push(self, index):
        self._stack.append(index)

    def pop(self):
        return self._stack.pop()

    def add_arc(self, index, head, label):
        self._heads[index] = head
        self._labels[index] = label

    def record(self, action):
        self._history.append(action)

    @property
    def step(self):
        return len(self._history)

    @property
    def num_tokens(self):
        return self._num_tokens

    def end(self):
        return self._buffer == self._num_tokens

    @property
    def stack_top(self):
        return self._stack[-1]

    def stack(self, position):
        if position < 0:
            return -1
        index = self.stack_size - 1 - position
        return -1 if index < 0 else self._stack[index]

    @property
    def stack_size(self):
        return len(self._stack)

    def statk_empty(self):
        return not(self._stack)

    @property
    def buffer_head(self):
        return self._buffer

    def buffer(self, position):
        if position < 0:
            return -1
        index = self._buffer + position
        return index if index < self._num_tokens else -1

    def head(self, index):
        return self._heads[index]

    @property
    def heads(self):
        return self._heads

    def label(self, index):
        return self._labels[index]

    @property
    def labels(self):
        return self._labels

    def leftmost(self, index, check_from=0):
        if (index >= 0 and index < self._num_tokens
                and check_from >= 0 and check_from < index):
            for i in range(check_from, index):
                if self._heads[i] == index:
                    return i
        return -1

    def rightmost(self, index, check_from=-1):
        check_from = self._num_tokens - 1 if check_from == - 1 else check_from
        if (index >= 0 and index < self._num_tokens
                and check_from > index and check_from < self._num_tokens):
            for i in range(check_from, index, -1):
                if self._heads[i] == index:
                    return i
        return -1

    def token(self, index):
        return self._sentence[index]

    def get_token(self, index, default):
        if index < 0 or index >= self._num_tokens:
            return default
        return self._sentence[index]

    @property
    def history(self):
        return self._history
