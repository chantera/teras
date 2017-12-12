"""
This library includes neural network models implemented with Chainer (v2.0.1)
"""

# import queue

from chainer import __version__ as chainer_version
from chainer import cuda, initializers, link, variable
import chainer.functions as F
import chainer.links as L
from chainer.links.connection.n_step_rnn import argsort_list_descent
from chainer.links.connection.n_step_rnn import permutate_list
import numpy as np

from teras.framework.chainer import functions as teras_F


if int(chainer_version[0]) > 2:
    batch_matmul = F.matmul
else:
    batch_matmul = F.batch_matmul


class EmbedID(link.Link):
    """same as chainer.links.EmbedID except for fixing pretrained weight"""

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None,
                 ignore_label=None, fixed_weight=False):
        super(EmbedID, self).__init__()
        self.ignore_label = ignore_label
        self.fixed_weight = fixed_weight

        with self.init_scope():
            if initialW is None:
                initialW = initializers.normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))
            if fixed_weight:
                self.W._requires_grad = False
                self.W._node._requires_grad = False

    def __call__(self, x):
        return teras_F.embed_id(x, self.W, self.ignore_label,
                                self.fixed_weight)


class Embed(link.ChainList):

    def __init__(self, *args, dropout=0.0):
        embeds = []
        self.size = 0
        for i, _args in enumerate(args):
            if isinstance(_args, dict):
                vocab_size = _args.get('in_size', None)
                embed_size = _args.get('out_size', None)
                embeddings = _args.get('initialW', None)
                if vocab_size is None or embed_size is None:
                    if embeddings is None:
                        raise ValueError('embeddings or in_size/out_size '
                                         'must be specified')
                    vocab_size, embed_size = embeddings.shape
                    _args['in_size'] = vocab_size
                    _args['out_size'] = embed_size
            else:
                if isinstance(_args, np.ndarray):
                    vocab_size, embed_size = _args.shape
                    embeddings = _args
                elif isinstance(_args, tuple) and len(embeddings) == 2:
                    vocab_size, embed_size = _args
                    embeddings = None
                else:
                    raise ValueError('embeddings must be '
                                     'np.ndarray or tuple(len=2)')
                _args = {'in_size': vocab_size, 'out_size': embed_size,
                         'initialW': embeddings}
            embeds.append(EmbedID(**_args))
            self.size += embed_size
        super(Embed, self).__init__(*embeds)

        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout

    def __call__(self, *xs):
        hs = []
        batch = len(xs[0])
        for i in range(batch):
            _hs = F.concat([teras_F.dropout(embed(self.xp.array(_xs[i])),
                                            self._dropout_ratio)
                            for _xs, embed in zip(xs, self)])
            hs.append(_hs)
        return hs


class MLP(link.ChainList):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        super(MLP, self).__init__(*layers)

    def __call__(self, x):
        for layer in self:
            x = layer(x)
        return x

    class Layer(L.Linear):

        def __init__(self, in_size, out_size=None,
                     activation=None, dropout=0.0,
                     nobias=False, initialW=None, initial_bias=None):
            super(MLP.Layer, self).__init__(in_size, out_size, nobias,
                                            initialW, initial_bias)
            if activation is None:
                self._activate = lambda x: x
            else:
                if not callable(activation):
                    raise ValueError("activation must be callable: type={}"
                                     .format(type(activation)))
                self._activate = activation
            assert dropout == 0 or type(dropout) == float
            self._dropout_ratio = dropout

        def __call__(self, x):
            shape = x.shape
            x = F.reshape(x, (-1, shape[-1]))
            y = super(MLP.Layer, self).__call__(x)
            y = F.reshape(y, shape[0:-1] + (-1,))
            return teras_F.dropout(self._activate(y), self._dropout_ratio)


class LSTM(L.NStepLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5,
                 initialW=None, initial_bias=None):
        if int(chainer_version[0]) > 2:
            super(LSTM, self).__init__(n_layers, in_size, out_size, dropout,
                                       initialW, initial_bias)
        else:
            super(LSTM, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx, cx = None, None
        hy, cy, ys = super(BiLSTM, self).__call__(hx, cx, xs)
        return ys


class BiLSTM(L.NStepBiLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5,
                 initialW=None, initial_bias=None):
        if int(chainer_version[0]) > 2:
            super(BiLSTM, self).__init__(n_layers, in_size, out_size, dropout,
                                         initialW, initial_bias)
        else:
            super(BiLSTM, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx, cx = None, None
        hy, cy, ys = super(BiLSTM, self).__call__(hx, cx, xs)
        return ys


class GRU(L.NStepGRU):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5,
                 initialW=None, initial_bias=None):
        if int(chainer_version[0]) > 2:
            super(GRU, self).__init__(n_layers, in_size, out_size, dropout,
                                      initialW, initial_bias)
        else:
            super(GRU, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx = None
        hy, ys = super(BiGRU, self).__call__(hx, xs)
        return ys


class BiGRU(L.NStepBiGRU):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5,
                 initialW=None, initial_bias=None):
        if int(chainer_version[0]) > 2:
            super(BiGRU, self).__init__(n_layers, in_size, out_size, dropout,
                                        initialW, initial_bias)
        else:
            super(BiGRU, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx = None
        hy, ys = super(BiGRU, self).__call__(hx, xs)
        return ys


class GlobalAttention(link.Chain):
    """This model has not been updated and tested for Chainer v2.0.0"""

    def __init__(self, n_units, score_func='general'):
        links = {}
        if score_func == 'dot':
            self._score_func = self._score_dot
        elif score_func == 'concat':
            links['W'] = L.Linear(n_units * 2, n_units * 2)
            links['v'] = L.Linear(n_units * 2, 1)
            self._score_func = self._score_concat
        else:
            links['W'] = L.Linear(n_units, n_units, nobias=True)
            self._score_func = self._score_general
        super(GlobalAttention, self).__init__(**links)

    def __call__(self, x, hs):
        batch, dim = x.shape
        alphas = 0
        _sum = 0
        for h in F.transpose_sequence(hs[:batch]):
            size = h.shape[0]
            if size < batch:
                h = F.vstack([h, variable.Variable(
                    self.xp.zeros((batch - size, h.shape[1]), dtype='f'))])
            score = self._score_func(x, h)
            e = F.exp(score)
            _sum += e
            alphas += batch_matmul(h, e)
        c = F.reshape(batch_matmul(F.reshape(alphas, (batch, dim)),
                                   (1 / _sum)), (batch, dim))
        return c

    def _score_general(self, x, h):
        batch, dim = x.shape
        return batch_matmul(F.reshape(self.W(x), (batch, 1, dim)), h)

    def _score_concat(self, x, h):
        return self.v(F.tanh(self.W(F.concat([x, h]))))

    def _score_dot(self, x, h):
        raise NotImplementedError()


class CRF(link.Chain):

    def __init__(self, n_label):
        super(CRF, self).__init__()
        with self.init_scope():
            self.cost = variable.Parameter(0, (n_label, n_label))

    def __call__(self, xs, ys, reduce='mean'):
        indices = argsort_list_descent(xs)
        xs = permutate_list(xs, indices, inv=False)
        xs = F.transpose_sequence(xs)
        ys = permutate_list(ys, indices, inv=False)
        ys = F.transpose_sequence(ys)
        return F.crf1d(self.cost, xs, ys, reduce)

    def argmax(self, xs):
        indices = argsort_list_descent(xs)
        xs = permutate_list(xs, indices, inv=False)
        xs = F.transpose_sequence(xs)
        score, path = F.argmax_crf1d(self.cost, xs)
        path = F.transpose_sequence(path)
        path = permutate_list(path, indices, inv=True)
        score = F.permutate(score, indices, inv=True)
        return score, path

    # def argnmax(self, xs, n=10):
    #     cost = cuda.to_cpu(self.cost.data)
    #     xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
    #     xs = [cuda.to_cpu(x.data) for x in xs]
    #
    #     scores = []
    #     paths = []
    #
    #     for _xs in xs:
    #         alphas = [_xs[0]]
    #         for x in _xs[1:]:
    #             alpha = np.max(alphas[-1] + cost, axis=1) + x
    #             alphas.append(alpha)
    #
    #         _scores = []
    #         _paths = []
    #         _end = len(_xs) - 1
    #         buf = n
    #
    #         c = queue.PriorityQueue()
    #         q = queue.PriorityQueue()
    #         x = _xs[_end]
    #         for i in range(x.shape[0]):
    #             q.put((-alphas[_end][i], -x[i], _end,
    #                    np.random.random(), np.array([i], np.int32)))
    #         while not q.empty() and c.qsize() < n + buf:
    #             beta, score, time, r, path = q.get()
    #             if time == 0:
    #                 c.put((score, r, path))
    #                 continue
    #             t = time - 1
    #             x = _xs[t]
    #             for i in range(x.shape[0]):
    #                 _trans = score - cost[i, path[-1]]
    #                 _beta = -alphas[t][i] + _trans
    #                 _score = _trans - x[i]
    #                 q.put((_beta, _score, t,
    #                        np.random.random(), np.append(path, i)))
    #         while not c.empty() and len(_paths) < n:
    #             score, r, path = c.get()
    #             _scores.append(-score)
    #             _paths.append(path[::-1])
    #         scores.append(_scores)
    #         paths.append(_paths)
    #
    #     return scores, paths


class CharCNN(link.Chain):

    def __init__(self, char_embeddings, pad_id,
                 out_size=50, window_size=3, dropout=0.5,
                 nobias=False, initialW=None, initial_bias=None):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd value: '{}' is given"
                             .format(window_size))
        super(CharCNN, self).__init__()
        char_vocab_size, char_embed_size = char_embeddings.shape
        with self.init_scope():
            self.embed = L.EmbedID(
                in_size=char_vocab_size,
                out_size=char_embed_size,
                initialW=char_embeddings,
            )
            self.conv = L.Convolution2D(
                in_channels=1,
                out_channels=out_size,
                ksize=(window_size, char_embed_size),
                stride=(1, char_embed_size),
                pad=(window_size // 2, 0),
                nobias=nobias,
                initialW=initialW,
                initial_bias=initial_bias
            )
        self.out_size = out_size
        self._pad_id = pad_id
        self._padding = np.array([pad_id] * (window_size // 2),
                                 dtype=np.int32)
        self._dropout = dropout

    def __call__(self, chars):
        if not isinstance(chars, (tuple, list)):
            chars = [chars]
        char_ids, boundaries = self._create_sequence(chars)
        x = self.embed(self.xp.array(char_ids))
        x = F.dropout(x, self._dropout)
        length, dim = x.shape
        C = self.conv(F.reshape(x, (1, 1, length, dim)))
        # C.shape -> (1, out_size, length, 1)
        C = F.split_axis(F.transpose(F.reshape(C, (self.out_size, length))),
                         boundaries, axis=0)
        ys = F.max(F.pad_sequence(
            [matrix for i, matrix in enumerate(C) if i % 2 == 1],
            padding=-self.xp.inf), axis=1)  # max over time pooling
        # assert len(chars) == ys.shape[0]
        return ys

    def _create_sequence(self, chars):
        char_ids = [self._padding]  # pad <BEGIN_OF_WORD>
        boundary = len(self._padding)
        boundaries = [boundary]
        pad_2w = np.concatenate((self._padding, self._padding))
        pad_2w_length = len(pad_2w)
        for _chars in chars[:-1]:
            char_ids.append(_chars)
            boundary += len(_chars)
            boundaries.append(boundary)
            char_ids.append(pad_2w)  # pad <END_OF_WORD> and <BEGIN_OF_WORD>
            boundary += pad_2w_length
            boundaries.append(boundary)
        char_ids.append(chars[-1])
        boundary += len(chars[-1])
        boundaries.append(boundary)
        char_ids.append(self._padding)  # pad <END_OF_WORD>
        char_ids = np.concatenate(char_ids)
        return char_ids, boundaries


class Biaffine(link.Link):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/linalg.py#L116  # NOQA
    """

    def __init__(self, left_size, right_size, out_size,
                 nobias=(False, False, False),
                 initialW=None, initial_bias=None):
        super(Biaffine, self).__init__()
        self.in_sizes = (left_size, right_size)
        self.out_size = out_size
        self.nobias = nobias

        with self.init_scope():
            shape = (left_size + int(not(self.nobias[0])),
                     right_size + int(not(self.nobias[1])),
                     out_size)
            if isinstance(initialW, (np.ndarray, cuda.ndarray)):
                assert initialW.shape == shape
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), shape)
            if not self.nobias[2]:
                if initial_bias is None:
                    initial_bias = 0
                self.b = variable.Parameter(initial_bias, (self.out_size,))
            else:
                self.b = None

    def __call__(self, x1, x2):
        xp = self.xp
        out_size = self.out_size
        batch_size, len1, dim1 = x1.shape
        if not self.nobias[0]:
            x1 = F.concat((x1, xp.ones((batch_size, len1, 1),
                                       dtype=xp.float32)), axis=2)
            dim1 += 1
        len2, dim2 = x2.shape[1:]
        if not self.nobias[1]:
            x2 = F.concat((x2, xp.ones((batch_size, len2, 1),
                                       dtype=xp.float32)), axis=2)
            dim2 += 1
        x1_reshaped = F.reshape(x1, (batch_size * len1, dim1))
        W_reshaped = F.reshape(F.transpose(self.W, (0, 2, 1)),
                               (dim1, out_size * dim2))
        affine = F.reshape(F.matmul(x1_reshaped, W_reshaped),
                           (batch_size, len1 * out_size, dim2))
        biaffine = F.transpose(
            F.reshape(batch_matmul(affine, x2, transb=True),
                      (batch_size, len1, out_size, len2)),
            (0, 1, 3, 2))
        if not self.nobias[2]:
            biaffine += F.broadcast_to(self.b, biaffine.shape)
        return biaffine
