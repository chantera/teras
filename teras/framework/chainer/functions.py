from chainer import __version__ as chainer_version
from chainer import configuration
import chainer.functions as F


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


if chainer_version.startswith('3'):
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
        return F.noise.dropout.Dropout(ratio)(x)
    return x
