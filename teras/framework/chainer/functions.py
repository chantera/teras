from chainer import __version__ as chainer_version
from chainer import as_variable
from chainer import configuration
from chainer import cuda
from chainer import function_node
import chainer.functions as F
# from chainer.utils import type_check


class EmbedIDFunction(
        F.connection.embed_id.EmbedIDFunction):

    def __init__(self, ignore_label=None, fixed_weight=False):
        super(EmbedIDFunction, self).__init__(ignore_label)
        self.fixed_weight = fixed_weight

    def backward(self, inputs, grad_outputs):
        if self.fixed_weight:
            """W does not require its gradient"""
            return None, None
        return super(EmbedIDFunction).backward(inputs, grad_outputs)


if int(chainer_version[0]) > 2:
    def embed_id(x, W, ignore_label=None, fixed_weight=False):
        if fixed_weight:
            return EmbedIDFunction(ignore_label, fixed_weight) \
                .apply((x, W))[0]
        else:
            return F.connection.embed_id.EmbedIDFunction(ignore_label) \
                .apply((x, W))[0]
else:
    def embed_id(x, W, ignore_label=None, fixed_weight=False):
        if fixed_weight:
            return EmbedIDFunction(ignore_label, fixed_weight)(x, W)
        else:
            return F.connection.embed_id.EmbedIDFunction(ignore_label)(x, W)


def dropout(x, ratio=.5):
    if configuration.config.train and ratio > .0:
        return F.noise.dropout.Dropout(ratio).apply((x,))[0]
    return as_variable(x)


class Roll(function_node.FunctionNode):

    """Roll array elements along a given axis."""

    def __init__(self, shift, axis=None):
        # @TODO: type check
        self.shift = shift
        self.axis = axis

    def check_type_forward(self, in_types):
        # @TODO: type check
        pass

    def forward(self, inputs):
        self.retain_inputs(())
        xp = cuda.get_array_module(*inputs)
        return xp.roll(inputs[0], self.shift, self.axis),

    def backward(self, indexes, gy):
        if isinstance(self.shift, (list, tuple)):
            shift = [-e for e in self.shift]
        else:
            shift = -self.shift

        return Roll(shift, self.axis).apply(gy)


def roll(x, shift, axis=None):
    return Roll(shift, axis).apply((x,))[0]
