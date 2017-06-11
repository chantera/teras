import chainer

from progressbar import ProgressBar

# from teras.app import App, arg, Trainer
from teras.app import App, arg
from teras import logging


def test(a):
    pass


class Resource(object):
    pass


# class Trainer(teras.app.Trainer):
#
#     def _process_train(self, data):
#         size = len(train_dataset)
#         batch_count = 0
#         loss = 0.0
#         accuracy = 0.0
#         p = ProgressBar(min_value=0, max_value=size, fd=sys.stderr).start()
#         for batch_index, batch in enumerate(train_dataset.batch(batch_size, colwise=True, shuffle=True)):
#             p.update((batch_size * batch_index) + 1)
#             batch_loss = self.
#         p.finish()
#
#     def _process_test(self, data):
#         size = len(train_dataset)
#         batch_count = 0
#         loss = 0.0
#         accuracy = 0.0
#         p = ProgressBar(min_value=0, max_value=size, fd=sys.stderr).start()
#         for batch_index, batch in enumerate(train_dataset.batch(batch_size, colwise=True, shuffle=True)):
#             p.update((batch_size * batch_index) + 1)
#             batch_loss = self.
#         p.finish()

#
# Resource.load_model()
# Resource.load_data()


# train = Trainer(lazy_loader=Resource)


def train(hoge, epoch):
    print(hoge, epoch)
    # train, test = chainer.datasets.get_mnist()
    #
    #
    # optimizer = None
    # trainer = Trainer(model, optimizer, loss, func)
    # trainer().fit()
    # score = trainer.evaluate()


def decode():
    pass


App.add_command('train', train, {
    'hoge': True,
    'epoch': arg('--epoch', type=int, default=20, help='a'),
}, description="execute train")
# App.add_arg('debug', True)
# App.add_arg('verbose', arg('--verbose', action='store_true', default=False))

App.add_command('decode', decode, {})

App.configure(loglevel=logging.TRACE)
App.run()
