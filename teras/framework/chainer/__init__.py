import chainer

from teras.framework.chainer import callbacks, model
from teras.training.trainer import TrainEvent

__all__ = ['callbacks', 'chainer_train_on', 'chainer_train_off', 'config',
           'model', 'set_debug', 'set_model_to_device', 'to_device']


# hack
chainer.Variable.__int__ = lambda self: int(self.data)
chainer.Variable.__float__ = lambda self: float(self.data)


def to_device(x, device=None):
    return chainer.dataset.convert.to_device(device, x)


def _update(optimizer, loss):
    optimizer.target.cleargrads()
    loss.backward()
    optimizer.update()


def chainer_train_on(*args, **kwargs):
    chainer.config.train = True
    chainer.config.enable_backprop = True


def chainer_train_off(*args, **kwargs):
    chainer.config.train = False
    chainer.config.enable_backprop = False


def set_debug(debug):
    if debug:
        chainer.config.debug = True
        chainer.config.type_check = True
    else:
        chainer.config.debug = False
        chainer.config.type_check = False


def set_model_to_device(model, device_id=-1):
    if device_id >= -1:
        chainer.cuda.get_device_from_id(device_id).use()
        model.to_gpu()
    else:
        model.to_cpu()


set_debug(chainer.config.debug)
chainer.config.use_cudnn = 'auto'


config = {
    'update': _update,
    'hooks': {
        TrainEvent.EPOCH_TRAIN_BEGIN: chainer_train_on,
        TrainEvent.EPOCH_VALIDATE_BEGIN: chainer_train_off,
    },
    'callbacks': []
}
