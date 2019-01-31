from configparser import ParsingError
import os
import signal
import sys

from teras import logging, utils
from teras.app.argparse import arg, ArgParser, ConfigArgParser
from teras.base import Singleton, classproperty, Context


class AppBase(Singleton):
    __instance = None
    _commands = {}
    _argparser = ArgParser()
    debug = False

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def add_command(cls, name, command, args={}, description=None):
        assert type(args) == dict
        cls._argparser.add_group(name, help=description)
        for arg_name, value in sorted(args.items(),
                                      key=lambda x: x[1].names[0]):
            cls.add_arg(arg_name, value, group=name)
        cls._commands[name] = command

    @classmethod
    def add_arg(cls, name, value, group=None):
        cls._argparser.add_arg(name, value, group)

    @classmethod
    def configure(cls, **kwargs):
        pass

    @classmethod
    def _parse_args(cls, command=None):
        return cls._argparser.parse(command=command)

    @classmethod
    def run(cls, command=None):
        cls.configure()
        command, command_args, common_args \
            = cls._parse_args(command)
        try:
            def handler(signum, frame):
                raise SystemExit("Signal(%d) received: "
                                 "The program %s will be closed"
                                 % (signum, __file__))
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
            os.umask(0)

            self = cls._get_instance()
            self._initialize(command, command_args, common_args)
            self._preprocess()
            self._process()
            self._postprocess()
            self._finalize()
        except Exception:
            logging.e("Exception occurred during execution:",
                      exc_info=True, stack_info=cls.debug)
        except SystemExit as e:
            logging.w(e)
        finally:
            logging.getLogger().finalize()
            # sys.exit(0)

    @classmethod
    def _get_instance(cls):
        if not cls._has_instance():
            instance = object.__new__(cls)
            instance._initialized = False
            instance._command = None
            instance._command_args = None
            instance._config = None
            instance._context = None
            cls.__instance = instance
        return cls.__instance

    @classmethod
    def _has_instance(cls):
        return cls.__instance is not None

    def _initialize(self, command, args, config):
        if self._initialized:
            return
        self._command = AppBase._commands[command]
        self._command_args = args
        self._config = config
        self._context = Context(**args)
        self._initialized = True

    def _preprocess(self):
        pass

    def _process(self):
        kwargs = self._command_args
        logging.d("App._process(self) called - command: {}, args: {}"
                  .format(self._command, kwargs))
        self._command(**kwargs)

    def _postprocess(self):
        pass

    def _finalize(self):
        pass


class App(AppBase):
    DEFAULT_APP_NAME = "teras.app"
    DEFAULT_CONFIG_FILE = "~/.teras.conf"
    app_name = ''
    _argparser = ConfigArgParser(DEFAULT_CONFIG_FILE)
    verbose = True

    @staticmethod
    def _static_initialize():
        if hasattr(App, '_static_initialized') and App._static_initialized:
            return
        entry_script = sys.argv[0]
        basedir = os.path.dirname(os.path.realpath(entry_script))
        entry_point = os.path.join(basedir, os.path.basename(entry_script))

        App.basedir = basedir
        App.entry_script = entry_script
        App.entry_point = entry_point

        repo = utils.git.root(entry_point)
        if repo is not None:
            App.app_name = os.path.basename(repo) + entry_point[len(repo):]
        else:
            App.app_name = App.DEFAULT_APP_NAME

        App._static_initialized = True

    @classmethod
    def configure(cls, **kwargs):
        if hasattr(cls, '_configured') and cls._configured:
            return
        if 'name' in kwargs:
            cls.app_name = kwargs['name']
        default_logdir = cls.basedir + '/logs'
        loglevel_choices = [logging.getLevelName(level) for level in
                            [logging.FATAL,
                             logging.WARN,
                             logging.INFO,
                             logging.DEBUG,
                             logging.TRACE]]
        cls.add_arg('debug', arg('--debug',
                                 action='store_true',
                                 default=kwargs.get('debug', False),
                                 help='Enable debug mode'))
        cls.add_arg('logdir', arg('--logdir',
                                  type=str,
                                  default=kwargs.get('logdir', default_logdir),
                                  help='Log directory',
                                  metavar='DIR'))
        cls.add_arg('loglevel', arg('--loglevel',
                                    type=str,
                                    default=kwargs.get('loglevel', 'info'),
                                    help='Log level',
                                    choices=loglevel_choices))
        cls.add_arg('logoption', arg('--logoption',
                                     type=str,
                                     default=kwargs.get('logoption', 'a'),
                                     help='Log option: {a,d,h,n,w}',
                                     metavar='VALUE'))
        cls.add_arg('quiet', arg('--quiet',
                                 action='store_true',
                                 default=kwargs.get('quiet', not(App.verbose)),
                                 help='execute quietly: '
                                 'does not print any message'))
        cls._configured = True

    @classmethod
    def _parse_args(cls, command=None):
        try:
            return cls._argparser.parse(command=command,
                                        section_prefix=cls.app_name)
        except (FileNotFoundError, ParsingError) as e:
            print(e, file=sys.stderr)
            sys.exit(0)

    def _preprocess(self):
        uname = os.uname()
        App.verbose = not(self._config['quiet'])
        App.debug = self._config['debug']

        loglevel = logging.logging._checkLevel(self._config['loglevel'])
        if App.debug:
            if (loglevel < logging.DISABLE
                    and loglevel > logging.DEBUG):
                loglevel = self._config['loglevel'] = logging.DEBUG
        logger_name = App.entry_script
        logger_config = {
            'logdir': self._config['logdir'],
            'filemode': 'a',
            'level': loglevel,
            'verbosity': logging.TRACE if App.verbose else logging.DISABLE
        }
        for char in self._config['logoption']:
            if char == 'a':
                logger_config['filemode'] = 'a'
            elif char == 'd':
                logger_config['filelog'] = False
            elif char == 'h':
                logger_config['filesuffix'] = '-' + uname[1]
            elif char == 'n':
                logger_config['filemode'] = 'n'
            elif char == 'w':
                logger_config['filemode'] = 'w'
            else:
                raise ValueError("Invalid logoption specified: {}"
                                 .format(char))
        logging.AppLogger.configure(**logger_config)
        logger = logging.AppLogger(logger_name)
        logging.setRootLogger(logger)
        if not App.verbose:
            sys.stdout = sys.stderr = open(os.devnull, 'w')

        logger.v(str(sys.version_info))
        logger.v(str(os.uname()))
        logger.i("sys.argv: %s" % str(sys.argv))
        logger.v("app._config: {}".format(self._config))
        logger.i("*** [START] ***")

    def _postprocess(self):
        logging.i("*** [DONE] ***")

    @classproperty
    def context(cls):
        if not cls._has_instance():
            return None
        return cls._get_instance()._context


App._static_initialize()
