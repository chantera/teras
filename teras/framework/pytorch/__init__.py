import torch

__all__ = ['config']


# hack
torch.autograd.Variable.__int__ = lambda self: int(self.data.cpu().numpy())
torch.autograd.Variable.__float__ = lambda self: float(self.data.cpu().numpy())


def _update(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


config = {
    'update': _update,
    'hooks': {},
    'callbacks': []
}
