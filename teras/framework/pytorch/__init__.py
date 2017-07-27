import torch

from teras.framework.pytorch import callbacks, model

__all__ = ['callbacks', 'config', 'model', 'set_model_to_device']


# hack
torch.autograd.Variable.__int__ = lambda self: int(self.data.cpu().numpy())
torch.autograd.Variable.__float__ = lambda self: float(self.data.cpu().numpy())


def _update(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def set_model_to_device(model, device_id=-1):
    if device_id >= 0 or device_id is None:
        model.cuda(device_id)
    else:
        model.cpu()


config = {
    'update': _update,
    'hooks': {},
    'callbacks': []
}
