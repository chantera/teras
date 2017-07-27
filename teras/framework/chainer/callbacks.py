import chainer

from teras.training.callbacks import Saver as _Saver
import teras.logging as Log


class Saver(_Saver):

    def __init__(self, model, basename, directory='', context=None,
                 name="chainer.saver", **kwargs):
        super(Saver, self).__init__(name, **kwargs)

    def on_epoch_end(self, data):
        model_file = "{}.{}.npz".format(self._basename, data['epoch'])
        Log.i("saving the model to {} ...".format(model_file))
        chainer.serializers.save_npz(model_file, self._model)
