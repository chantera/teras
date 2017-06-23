from enum import Enum
from datetime import datetime
import logging
import logging.config
import os
import sys
import time
import uuid

from dateutil.tz import tzlocal


DISABLE = sys.maxsize
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
TRACE = 5
NOTSET = logging.NOTSET

logging.addLevelName(DISABLE, 'disabled')
logging.addLevelName(CRITICAL, 'critical')
logging.addLevelName(FATAL, 'fatal')
logging.addLevelName(ERROR, 'error')
logging.addLevelName(WARNING, 'warning')
logging.addLevelName(WARN, 'warn')
logging.addLevelName(INFO, 'info')
logging.addLevelName(DEBUG, 'debug')
logging.addLevelName(TRACE, 'trace')
logging.addLevelName(NOTSET, 'none')


class Formatter(logging.Formatter):

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            t = datefmt.replace('%f', '{:03d}'.format(int(record.msecs)))
            s = time.strftime(t, ct)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s


class Color(int, Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7


class ColoredFormatter(Formatter):
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[%dm"
    FORMAT = COLOR_SEQ + "%s" + RESET_SEQ
    COLORS = {
        CRITICAL: Color.RED,
        FATAL: Color.RED,
        ERROR: Color.RED,
        WARNING: Color.YELLOW,
        WARN: Color.YELLOW,
        INFO: Color.WHITE,
        DEBUG: Color.CYAN,
        TRACE: Color.CYAN,
    }

    def format(self, record):
        s = super(ColoredFormatter, self).format(record)
        level = record.levelno
        if level in ColoredFormatter.COLORS:
            s = (ColoredFormatter.FORMAT
                 % (30 + ColoredFormatter.COLORS[level], s))
        return s


class Logger(logging.Logger):

    def __init__(self, name, level=NOTSET, handlers=[]):
        self._initialized = False
        super(Logger, self).__init__(name, level)
        for hdlr in handlers:
            self.addHandler(hdlr)
        self.initialize()

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG):
            self._log(TRACE, msg, args, **kwargs)

    e = logging.Logger.error
    w = logging.Logger.warning
    i = logging.Logger.info
    d = logging.Logger.debug
    v = trace

    def initialize(self):
        if self._initialized:
            return
        self._initialized = True

    def finalize(self):
        for hdlr in self.handlers:
            self.removeHandler(hdlr)
        self.disabled = True

    @property
    def initialized(self):
        return self._initialized


class RootLogger(Logger):

    def __init__(self, level):
        Logger.__init__(self, "root", level)


def setRootLogger(root):
    if not isinstance(root, Logger):
            raise TypeError("logger not derived from teras.logging.Logger: "
                            + type(root).__name__)
    logging.root = root
    Logger.root = root
    Logger.manager.root = root


logging.setLoggerClass(Logger)
setRootLogger(RootLogger(WARNING))


PATH_SEP = os.path.sep
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f %Z"


class AppLogger(Logger):
    FORMAT = "%(asctime)-15s\t%(accessid)s\t[%(levelname)s]\t%(message)s"
    _config = {
        'level': INFO,
        'verbosity': TRACE,
        'filelog': True,
        'logdir': None,
        'filename': "%Y%m%d.log",
        'filemode': 'a',
        'fileprefix': '',
        'filesuffix': '',
        'fmt': FORMAT,
        'datefmt': DATE_FORMAT,
    }

    @classmethod
    def configure(cls, **kwargs):
        cls._config.update(kwargs)

    def initialize(self):
        super(AppLogger, self).initialize()
        config = AppLogger._config
        now = datetime.now(tzlocal())
        self._accessid = uuid.uuid4().hex[:6]
        self._uniqueid = "UNIQID"
        self._accesssec = now
        self._accesstime = now.strftime(config['datefmt'])

        if len(self.handlers) == 0:
            if config['filelog']:
                self._add_file_handler(config)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(config['verbosity'])
            stream_handler.setFormatter(
                ColoredFormatter(config['fmt'], config['datefmt']))
            self.addHandler(stream_handler)

        message = "LOG Start with ACCESSID=[%s] UNIQUEID=[%s] ACCESSTIME=[%s]"
        self.info(message % (self._accessid, self._uniqueid, self._accesstime))

    def _add_file_handler(self, config):
        enable_numbering = False
        filemode = 'a'
        if config['filemode'] == '' or config['filemode'] == 'a':
            filemode = 'a'
        elif config['filemode'] == 'w':
            filemode = 'w'
        elif config['filemode'] == 'n':
            filemode = 'w'
            enable_numbering = True
        else:
            raise ValueError("Invalid filemode specified: {}"
                             .format(config['filemode']))

        logdir = config['logdir']
        if logdir:
            logdir = os.path.abspath(os.path.expanduser(logdir))
            if not os.path.isdir(logdir):
                raise FileNotFoundError("logdir was not found: "
                                        "'%s'" % logdir)
            logdir += PATH_SEP
        else:
            logdir = ''

        if PATH_SEP in config['filename']:
            raise ValueError("Invalid character '{}' is included: {}"
                             .format(PATH_SEP, config['filename']))

        basename, ext = os.path.splitext(config['filename'])
        basename = (config['fileprefix']
                    + datetime.now().strftime(basename)
                    + config['filesuffix'])

        if enable_numbering:
            number = 0
            while True:
                logfile = logdir + basename + '-' + str(number) + ext
                if not os.path.exists(logfile):
                    break
                number += 1
        else:
            logfile = logdir + basename + ext

        file_handler = logging.FileHandler(logfile, mode=filemode)
        file_handler.setLevel(config['level'])
        file_handler.setFormatter(
            Formatter(config['fmt'], config['datefmt']))

        self.addHandler(file_handler)

    def finalize(self):
        processtime = ('%3.9f' % (datetime.now(tzlocal())
                                  - self._accesssec).total_seconds())
        message = ("LOG End with ACCESSID=[%s] UNIQUEID=[%s] "
                   "ACCESSTIME=[%s] PROCESSTIME=[%s]\n")
        self.info(message % (self._accessid, self._uniqueid,
                             self._accesstime, processtime))
        super(AppLogger, self).finalize()

    def filter(self, record):
        record.accessid = self._accessid
        return super(AppLogger, self).filter(record)

    @property
    def accessid(self):
        return self._accessid

    @property
    def accesstime(self):
        return self._accesssec


BASIC_FORMAT = logging.BASIC_FORMAT
APP_FORMAT = AppLogger.FORMAT


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'basic': {
            '()': Formatter,
            'format': BASIC_FORMAT,
            'datefmt': DATE_FORMAT,
        },
        'color': {
            '()': ColoredFormatter,
            'format': BASIC_FORMAT,
            'datefmt': DATE_FORMAT,
        },
    },
    'handlers': {
        'color': {
            'class': 'logging.StreamHandler',
            'formatter': 'color',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['color'],
    }
}

logging.config.dictConfig(LOGGING)


def trace(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        logging.basicConfig()
    logging.root.trace(msg, *args, **kwargs)


e = logging.error
w = logging.warning
i = logging.info
d = logging.debug
v = trace


for module in logging.__all__:
    if module not in globals():
        globals()[module] = getattr(globals()['logging'], module)
