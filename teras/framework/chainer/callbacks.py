import chainer

from teras.training.callbacks import Saver as _Saver
import teras.logging as Log


class Saver(_Saver):

    def __init__(self, model, basename, directory='', context=None,
                 interval=1, save_from=None, name="chainer.saver", **kwargs):
        super(Saver, self).__init__(model, basename, directory, context,
                                    interval, save_from, name, **kwargs)

    def on_epoch_end(self, data):
        epoch = data['epoch']
        if self._save_from is not None and data['epoch'] < self._save_from:
            return
        if epoch % self._interval == 0:
            model_file = "{}.{}.npz".format(self._basename, epoch)
            Log.i("saving the model to {} ...".format(model_file))
            chainer.serializers.save_npz(model_file, self._model)
