from abc import abstractmethod
from collections.abc import Callable

import dynet


class Network(Callable):

    @abstractmethod
    def init_params(self, model):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        return self.__call__(args, kwargs)


class MLP(Network):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        self._layers = layers

    def init_params(self, model):
        for layer in self._layers:
            layer.init_params(model)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    class Layer(Network):

        def __init__(self, in_size, out_size,
                     activation=None, dropout=0.0):
            self.W = None
            self.b = None
            self.in_size = in_size
            self.out_size = out_size
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
                self._dropout_func = lambda x, ratio: dynet.dropout(x, ratio)
            else:
                self._dropout_func = lambda x, ratio: x

        def init_params(self, model):
            self.W = model.add_parameters((self.out_size, self.in_size))
            self.b = model.add_parameters((self.out_size,))

        def __call__(self, x):
            W = dynet.parameter(self.W)
            b = dynet.parameter(self.b)
            y = dynet.affine_transform([b, W, x])
            return self._dropout_func(self._activate(y), self._dropout_ratio)
