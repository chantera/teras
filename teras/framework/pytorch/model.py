import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class Embed(nn.ModuleList):

    def __init__(self, *args, dropout=0.0, padding_idx=None):
        embeds = []
        self.size = 0
        for i, embeddings in enumerate(args):
            if type(embeddings) is np.ndarray:
                vocab_size, embed_size = embeddings.shape
            elif type(embeddings) is tuple and len(embeddings) == 2:
                vocab_size, embed_size = embeddings
                embeddings = None
            else:
                raise ValueError('embeddings must be '
                                 'np.ndarray or tuple(len=2)')
            embed = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_size,
                padding_idx=padding_idx,
            )
            if embeddings is not None:
                embed.weight.data.copy_(torch.from_numpy(embeddings))
            embeds.append(embed)
            self.size += embed_size
        super(Embed, self).__init__(embeds)

        assert dropout == 0 or type(dropout) == float
        self._dropout_ratio = dropout
        if dropout > 0:
            self._dropout = nn.Dropout(p=self._dropout_ratio)
        else:
            self._dropout = lambda x: x

    def __call__(self, *xs):
        hs = []
        batch = len(xs[0])
        for i in range(batch):
            _hs = torch.cat(
                [self._dropout(
                    embed(Variable(torch.from_numpy(_xs[i].astype(np.int64)))))
                 for _xs, embed in zip(xs, self)], dim=1)
            hs.append(_hs)
        return hs


class MLP(nn.ModuleList):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        super(MLP, self).__init__(layers)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    class Layer(nn.Linear):

        def __init__(self, in_features, out_features,
                     activation=None, dropout=0.0, bias=True):
            super(MLP.Layer, self).__init__(in_features, out_features, bias)
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
                self._dropout = nn.Dropout(p=self._dropout_ratio)
            else:
                self._dropout = lambda x: x

        def forward(self, x):
            size = x.size()
            if len(size) > 2:
                x.contiguous()
                y = super(MLP.Layer, self).forward(x.view(-1, size[-1]))
                y.view(size[0:-1] + (-1,))
            else:
                y = super(MLP.Layer, self).forward(x)
            return self._dropout(self._activate(y))
