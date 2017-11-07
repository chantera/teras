from chainer import cuda
from chainer.initializer import Initializer


class Orthonormal(Initializer):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/linalg.py#L35  # NOQA
    """
    _logger = None

    def __init__(self, dtype=None):
        if self._logger is None:
            import logging
            Orthonormal._logger = logging.getLogger()
        super(Orthonormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        output_size, input_size = array.shape
        I = xp.eye(output_size)
        lr = .1
        eps = .05 / (output_size + input_size)
        success = False
        tries = 0
        while not success and tries < 10:
            Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
            for i in range(100):
                QTQmI = Q.T.dot(Q) - I
                loss = xp.sum(QTQmI ** 2 / 2)
                Q2 = Q ** 2
                Q -= lr * Q.dot(QTQmI) / \
                    (xp.abs(Q2 + Q2.sum(axis=0, keepdims=True)
                            + Q2.sum(axis=1, keepdims=True) - 1) + eps)
                if xp.max(Q) > 1e6 or loss > 1e6 or not xp.isfinite(loss):
                    tries += 1
                    lr /= 2
                    break
            success = True
        if success:
            self._logger.trace('Orthogonal pretrainer loss: %.2e' % loss)
        else:
            self._logger.trace('Orthogonal pretrainer failed, '
                               'using non-orthogonal random matrix')
        Q = xp.random.randn(input_size, output_size) / xp.sqrt(output_size)
        array[...] = Q.T
