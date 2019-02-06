from teras.training import event  # NOQA
from teras.training import listeners  # NOQA
from teras.training.listeners import report  # NOQA
from teras.training.trainer import Trainer  # NOQA


TRAIN_BEGIN = event.TrainEvent.TRAIN_BEGIN
TRAIN_END = event.TrainEvent.TRAIN_END
EPOCH_BEGIN = event.TrainEvent.EPOCH_BEGIN
EPOCH_END = event.TrainEvent.EPOCH_END
EPOCH_TRAIN_BEGIN = event.TrainEvent.EPOCH_TRAIN_BEGIN
EPOCH_TRAIN_END = event.TrainEvent.EPOCH_TRAIN_END
EPOCH_VALIDATE_BEGIN = event.TrainEvent.EPOCH_VALIDATE_BEGIN
EPOCH_VALIDATE_END = event.TrainEvent.EPOCH_VALIDATE_END
BATCH_BEGIN = event.TrainEvent.BATCH_BEGIN
BATCH_END = event.TrainEvent.BATCH_END
