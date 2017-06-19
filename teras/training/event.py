from teras.base.event import Event


class TrainEvent(Event):
    TRAIN_BEGIN = 'train_begin'
    TRAIN_END = 'train_end'
    EPOCH_BEGIN = 'epoch_begin'
    EPOCH_END = 'epoch_end'
    EPOCH_TRAIN_BEGIN = 'epoch_train_begin'
    EPOCH_TRAIN_END = 'epoch_train_end'
    EPOCH_VALIDATE_BEGIN = 'epoch_validate_begin'
    EPOCH_VALIDATE_END = 'epoch_validate_end'
    BATCH_BEGIN = 'batch_begin'
    BATCH_END = 'batch_end'
