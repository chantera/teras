#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L
from chainer.links.connection.n_step_lstm import argsort_list_descent, permutate_list
import numpy as np


class Embed(ChainList):

    def __init__(self, *args):
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

    def __call__(self, *xs):
        hs = []
        batch = len(xs[0])
        for i in range(batch):
            _hs = F.concat([embed(self.xp.array(_xs[i])) for _xs, embed in zip(xs, self)])
            hs.append(_hs)
        return hs


class LSTM(L.NStepLSTM):

    def __init__(self, in_size, out_size, dropout=0.5, use_cudnn=True):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')

        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys


class BLSTM(Chain):

    def __init__(self, n_units, dropout=0.5, train=True):
        super(BLSTM, self).__init__(
            f_lstm=LSTM(n_units, n_units, dropout),
            b_lstm=LSTM(n_units, n_units, dropout),
        )
        self._dropout = dropout
        self._n_units = n_units
        self.train = train

    def __call__(self, xs):
        self.f_lstm.reset_state()
        self.b_lstm.reset_state()
        xs_f = []
        xs_b = []
        for x in xs:
            xs_f.append(x)
            xs_b.append(x[::-1])
        hs_f = self.f_lstm(xs_f, self.train)
        hs_b = self.b_lstm(xs_b, self.train)
        ys = [F.concat([h_f, h_b[::-1]]) for h_f, h_b in zip(hs_f, hs_b)]
        return ys


class GlobalAttention(Chain):

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
                h = F.vstack([h, Variable(self.xp.zeros((batch - size, h.shape[1]), dtype='f'))])
            score = self._score_func(x, h)
            e = F.exp(score)
            _sum += e
            alphas += F.batch_matmul(h, e)
        c = F.reshape(F.batch_matmul(F.reshape(alphas, (batch, dim)), (1 / _sum)), (batch, dim))
        return c

    def _score_general(self, x, h):
        batch, dim = x.shape
        return F.batch_matmul(F.reshape(self.W(x), (batch, 1, dim)), h)

    def _score_concat(self, x, h):
        return self.v(F.tanh(self.W(F.concat([x, h]))))

    def _score_dot(self, x, h):
        raise NotImplementedError()


class CRF(L.CRF1d):

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
