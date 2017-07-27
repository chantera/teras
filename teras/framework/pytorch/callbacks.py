import torch

from teras.training.callbacks import Saver as _Saver
import teras.logging as Log


class Saver(_Saver):

    def __init__(self, model, basename, directory='', context=None,
                 name="pytorch.saver", **kwargs):
        super(Saver, self).__init__(name, **kwargs)

    def on_epoch_end(self, data):
        model_file = "{}.{}.mdl".format(self._basename, data['epoch'])
        Log.i("saving the model to {} ...".format(model_file))
        torch.save(self._model.state_dict(), model_file)
