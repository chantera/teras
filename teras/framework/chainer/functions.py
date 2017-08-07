import chainer.functions as F


class EmbedIDFunction(
        F.connection.embed_id.EmbedIDFunction):

    def __init__(self, ignore_label=None, fixed_weight=False):
        super(EmbedIDFunction).__init__(ignore_label)
        self.fixed_weight = fixed_weight

    def backward(self, inputs, grad_outputs):
        if self.fixed_weight:
            """W does not require its gradient"""
            return None, None
        return super(EmbedIDFunction).backward(inputs, grad_outputs)


def embed_id(x, W, ignore_label=None, fixed_weight=False):
    if fixed_weight:
        return EmbedIDFunction(ignore_label, fixed_weight)(x, W)
    else:
        return F.connection.embed_id.EmbedIDFunction(ignore_label)(x, W)
