from enum import Enum
from datetime import datetime
import logging
import logging.config
import os
import sys
import time
import uuid
import warnings


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

BASIC_FORMAT = logging.BASIC_FORMAT
APP_FORMAT = "%(asctime)-15s\t%(accessid)s\t[%(levelname)s]\t%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f %Z"


def _format_time(format, t, nsecs=None, precision=6):
    if '%f' in format:
        if nsecs is None:
            if isinstance(t, float):
                nsecs = int((t - int(t)) * 1e+6)
            else:
                nsecs = 0
        if precision < 6:
            nsecs = int(nsecs * (0.1 ** (6 - precision)))
        format = format.replace(
            '%f', '{:0{prec}d}'.format(nsecs, prec=precision))
    return time.strftime(format, t)


class Formatter(logging.Formatter):

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = _format_time(datefmt, ct, int(record.msecs * 1000), 3)
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
        s = super().format(record)
        level = record.levelno
        if level in ColoredFormatter.COLORS:
            s = (ColoredFormatter.FORMAT
                 % (30 + ColoredFormatter.COLORS[level], s))
        return s


class Logger(logging.Logger):

    def __init__(self, name, level=NOTSET, handlers=[]):
        self._initialized = False
        super().__init__(name, level)
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


class AppLogger(Logger):
    _config = {
        'level': INFO,
        'verbosity': TRACE,
        'filelog': True,
        'logdir': None,
        'filename': "%Y%m%d.log",
        'filemode': 'a',
        'fileprefix': '',
        'filesuffix': '',
        'fmt': APP_FORMAT,
        'datefmt': DATE_FORMAT,
        'mkdir': False,
    }

    @classmethod
    def configure(cls, **kwargs):
        cls._config.update(kwargs)

    def initialize(self):
        super().initialize()
        config = AppLogger._config
        now = time.time()
        self._accessid = uuid.uuid4().hex[:6]
        self._uniqueid = "UNIQID"
        self._accesssec = now
        self._accesstime = _format_time(
            config['datefmt'], logging.Formatter.converter(now),
            nsecs=int((now - int(now)) * 1e+6))

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

        logfile = self._resolve_file(config, enable_numbering)
        file_handler = logging.FileHandler(logfile, mode=filemode)
        file_handler.setLevel(config['level'])
        file_handler.setFormatter(
            Formatter(config['fmt'], config['datefmt']))

        self.addHandler(file_handler)

    def _resolve_file(self, config, enable_numbering=False):
        logdir = config['logdir']
        if logdir:
            logdir = os.path.abspath(os.path.expanduser(logdir))
            if os.path.isdir(logdir):
                pass
            elif config['mkdir']:
                os.makedirs(logdir)
            else:
                raise FileNotFoundError("logdir was not found: `%s`" % logdir)
        else:
            logdir = ''

        if os.path.sep in config['filename']:
            raise ValueError("Invalid character '{}' is included: {}"
                             .format(os.path.sep, config['filename']))

        basename, ext = os.path.splitext(config['filename'])
        basename = (config['fileprefix']
                    + datetime.now().strftime(basename)
                    + config['filesuffix'])

        if enable_numbering:
            number = 0
            while True:
                logfile = os.path.join(
                    logdir, basename + '-' + str(number) + ext)
                if not os.path.exists(logfile):
                    break
                number += 1
        else:
            logfile = os.path.join(logdir, basename + ext)

        return logfile

    def finalize(self):
        processtime = ('%3.9f' % (time.time() - self._accesssec))
        message = ("LOG End with ACCESSID=[%s] UNIQUEID=[%s] "
                   "ACCESSTIME=[%s] PROCESSTIME=[%s]\n")
        self.info(message % (self._accessid, self._uniqueid,
                             self._accesstime, processtime))
        super().finalize()

    def filter(self, record):
        record.accessid = self._accessid
        return super().filter(record)

    @property
    def accessid(self):
        return self._accessid

    @property
    def accesstime(self):
        return self._accesssec


DEFAULT_CONFIG = {
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
        'level': 'WARNING',
        'handlers': ['color'],
    }
}


def defaultConfig():
    if len(logging.root.handlers) == 0:
        logging.config.dictConfig(DEFAULT_CONFIG)


def critical(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.critical(msg, *args, **kwargs)


fatal = critical


def error(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    error(msg, *args, exc_info=exc_info, **kwargs)


def warning(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.warning(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    warnings.warn("The 'warn' function is deprecated, "
                  "use 'warning' instead", DeprecationWarning, 2)
    warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.debug(msg, *args, **kwargs)


def trace(msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.trace(msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
    if len(logging.root.handlers) == 0:
        defaultConfig()
    logging.root.log(level, msg, *args, **kwargs)


e = logging.error
w = logging.warning
i = logging.info
d = logging.debug
v = trace


logging.setLoggerClass(Logger)
setRootLogger(RootLogger(WARNING))

logging.getLogger(__name__.split('.')[0]).addHandler(logging.NullHandler())


for module in logging.__all__:
    if module not in globals():
        globals()[module] = getattr(globals()['logging'], module)
