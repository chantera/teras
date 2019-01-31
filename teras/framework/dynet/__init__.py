import dynet

from teras.framework.chainer import model
from teras.training.trainer import TrainEvent
from teras.utils.builtin import patches

__all__ = ['dynet_train_on', 'dynet_train_off', 'config', 'model']


_dynet_train = False


# hack
@patches(dynet.Expression, '__int__')
def __int__(self):
    return int(self.scalar_value())


@patches(dynet.Expression, '__float__')
def __float__(self):
    return self.scalar_value()


def _update(optimizer, loss):
    loss.backward()
    optimizer.update()


def dynet_train_on(*args, **kwargs):
    global _dynet_train
    _dynet_train = True


def dynet_train_off(*args, **kwargs):
    global _dynet_train
    _dynet_train = False


config = {
    'update': _update,
    'hooks': {
        TrainEvent.EPOCH_TRAIN_BEGIN: dynet_train_on,
        TrainEvent.EPOCH_VALIDATE_BEGIN: dynet_train_off,
        TrainEvent.BATCH_BEGIN: lambda x: dynet.renew_cg(),
    },
    'callbacks': []
}
