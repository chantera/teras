"""
This library includes neural network models implemented with Chainer (v2.0.0)
"""

import math
import queue

from chainer import Chain, ChainList, cuda, initializers, link, Variable
import chainer.functions as F
import chainer.links as L
from chainer.links.connection.n_step_rnn import (
    argsort_list_descent, permutate_list)
import numpy as np


class Embed(ChainList):

    def __init__(self, *args, **kwargs):
        embeds = []
        for i, embeddings in enumerate(args):
            vocab_size, embed_size = embeddings.shape
            embed = L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embeddings,
            )
            embeds.append(embed)
        super(Embed, self).__init__(*embeds)

        dropout = kwargs.get('dropout', 0.0)
        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout_func = lambda x, ratio: F.dropout(x, ratio)
        else:
            self._dropout_func = lambda x, ratio: x

    def __call__(self, *xs):
        hs = []
        batch = len(xs[0])
        for i in range(batch):
            _hs = F.concat([self._dropout_func(embed(self.xp.array(_xs[i])),
                                               self._dropout_ratio)
                            for _xs, embed in zip(xs, self)])
            hs.append(_hs)
        return hs


class MLP(ChainList):

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
            if dropout > 0:
                self._dropout_func = lambda x, ratio: F.dropout(x, ratio)
            else:
                self._dropout_func = lambda x, ratio: x

        def __call__(self, x):
            y = super(MLP.Layer, self).__call__(x)
            return self._dropout_func(self._activate(y), self._dropout_ratio)


class LSTM(L.NStepLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5):
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx, cx = None, None
        hy, cy, ys = super(BiLSTM, self).__call__(hx, cx, xs)
        return ys


class BiLSTM(L.NStepBiLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5):
        super(BiLSTM, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx, cx = None, None
        hy, cy, ys = super(BiLSTM, self).__call__(hx, cx, xs)
        return ys


class GRU(L.NStepGRU):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5):
        super(GRU, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx = None
        hy, ys = super(BiGRU, self).__call__(hx, xs)
        return ys


class BiGRU(L.NStepBiGRU):

    def __init__(self, n_layers, in_size, out_size, dropout=0.5):
        super(BiGRU, self).__init__(n_layers, in_size, out_size, dropout)

    def __call__(self, xs):
        hx = None
        hy, ys = super(BiGRU, self).__call__(hx, xs)
        return ys


class GlobalAttention(Chain):
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
                h = F.vstack([h, Variable(
                    self.xp.zeros((batch - size, h.shape[1]), dtype='f'))])
            score = self._score_func(x, h)
            e = F.exp(score)
            _sum += e
            alphas += F.batch_matmul(h, e)
        c = F.reshape(F.batch_matmul(F.reshape(alphas, (batch, dim)),
                                     (1 / _sum)), (batch, dim))
        return c

    def _score_general(self, x, h):
        batch, dim = x.shape
        return F.batch_matmul(F.reshape(self.W(x), (batch, 1, dim)), h)

    def _score_concat(self, x, h):
        return self.v(F.tanh(self.W(F.concat([x, h]))))

    def _score_dot(self, x, h):
        raise NotImplementedError()


class CRF(L.CRF1d):
    """This model has not been updated and tested for Chainer v2.0.0"""

    def __init__(self, n_label):
        super(CRF, self).__init__(n_label)

    def __call__(self, xs, ys):
        xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
        xs = F.transpose_sequence(xs)
        ys = permutate_list(ys, argsort_list_descent(ys), inv=False)
        ys = F.transpose_sequence(ys)
        return super(CRF, self).__call__(xs, ys)

    def argmax(self, xs):
        xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
        xs = F.transpose_sequence(xs)
        score, path = super(CRF, self).argmax(xs)
        path = F.transpose_sequence(path)
        return score, path

    def argnmax(self, xs, n=10):
        cost = cuda.to_cpu(self.cost.data)
        xs = permutate_list(xs, argsort_list_descent(xs), inv=False)
        xs = [cuda.to_cpu(x.data) for x in xs]

        scores = []
        paths = []

        for _xs in xs:
            alphas = [_xs[0]]
            for x in _xs[1:]:
                alpha = np.max(alphas[-1] + cost, axis=1) + x
                alphas.append(alpha)

            _scores = []
            _paths = []
            _end = len(_xs) - 1
            buf = n

            c = queue.PriorityQueue()
            q = queue.PriorityQueue()
            x = _xs[_end]
            for i in range(x.shape[0]):
                q.put((-alphas[_end][i], -x[i], _end,
                       np.random.random(), np.array([i], np.int32)))
            while not q.empty() and c.qsize() < n + buf:
                beta, score, time, r, path = q.get()
                if time == 0:
                    c.put((score, r, path))
                    continue
                t = time - 1
                x = _xs[t]
                for i in range(x.shape[0]):
                    _trans = score - cost[i, path[-1]]
                    _beta = -alphas[t][i] + _trans
                    _score = _trans - x[i]
                    q.put((_beta, _score, t,
                           np.random.random(), np.append(path, i)))
            while not c.empty() and len(_paths) < n:
                score, r, path = c.get()
                _scores.append(-score)
                _paths.append(path[::-1])
            scores.append(_scores)
            paths.append(_paths)

        return scores, paths


class CharCNN(Chain):

    def __init__(self, char_embeddings, window_size=3, dropout=0.5):
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
                out_channels=1,
                ksize=(1, window_size),
                stride=(1, 1),
                pad=(0, int(window_size / 2)),
                nobias=True,
                initialW=None,
            )
        self._dropout = dropout

    def __call__(self, chars):
        if type(chars) == tuple or type(chars) == list:
            return F.vstack([self.forward_one(_chars) for _chars in chars])
        return self.forward_one(chars)

    def forward_one(self, chars):
        x = self.embed(self.xp.array(chars))
        x = F.dropout(x, self._dropout)
        h, w = x.shape
        C = self.conv(F.reshape(x, (1, 1, h, w)))
        A = F.max_pooling_2d(C, ksize=(h, 1), stride=None, pad=0)
        y = F.reshape(A, (w,))
        return y


class Biaffine(link.Link):
    """This model has not been updated and tested for Chainer v2.0.0"""

    def __init__(self, in_size, out_size, wscale=1, initialW=None):
        super(Biaffine, self).__init__()
        self.initialW = initialW
        self.wscale = wscale
        self.out_size = out_size
        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))
        self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.add_param('W', (in_size + 1, self.out_size),
                       initializer=self._W_initializer)

    def __call__(self, x1, x2):
        """https://github.com/tdozat/Parser/blob/master/lib/linalg.py"""
        dim = len(x1.shape)
        if dim == 3:
            return self.forward_batch(x1, x2)
        elif dim == 2:
            return self.forward_one(x1, x2)
        else:
            raise RuntimeError()

    def forward_one(self, x1, x2):
        xp = cuda.get_array_module(x1.data)
        l, d = x2.shape
        return F.matmul(F.concat([x1, xp.ones((l, 1), 'f')]),
                        F.matmul(self.W, x2, transb=True))

    def forward_batch(self, x1, x2):
        xp = cuda.get_array_module(x1.data)
        b, l, d = x2.shape
        return F.batch_matmul(F.concat([x1, xp.ones((b, l, 1), 'f')], 2),
                              F.reshape(
                                  F.linear(F.reshape(x2, (b * l, -1)), self.W),
                                  (b, l, -1)), transb=True)
