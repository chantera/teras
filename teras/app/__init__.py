# from abc import abstractmethod
from argparse import ArgumentParser
from configparser import ConfigParser
import os
import re
import signal
import sys

from ..base import Singleton
from .. import logging


class _Args(object):

    class CmdlineArg(object):

        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs

        @property
        def args(self):
            return self._args

        @property
        def kwargs(self):
            return self._kwargs

        def add_arg(self, value):
            self._args.append(value)

        def add_kwarg(self, name, value):
            self._kwargs[name] = value

    def __init__(self):
        self._sys_argv = sys.argv
        self._common_cmd_args = {}
        self._common_const_args = {}
        self._groups = []
        self._group_cmd_args = {}
        self._group_const_args = {}
        self._group_descriptions = {}

    def def_arg(self, name, value):
        if type(value) == _Args.CmdlineArg:
            value.add_kwarg('dest', name)
            self._common_cmd_args[name] = value
        else:
            self._common_const_args[name] = value

    def def_group_arg(self, group, name, value):
        if group not in self._groups:
            self.add_group(group)
        if type(value) == _Args.CmdlineArg:
            value.add_kwarg('dest', name)
            self._group_cmd_args[group][name] = value
        else:
            self._group_const_args[group][name] = value

    def add_group(self, group, description=None):
        if group not in self._groups:
            self._groups.append(group)
            self._group_cmd_args[group] = {}
            self._group_const_args[group] = {}
            self._group_descriptions[group] = description

    @property
    def sys_argv(self):
        return self._sys_argv

    @property
    def common_cmd_args(self):
        return self._common_cmd_args

    @property
    def common_const_args(self):
        return self._common_const_args

    @property
    def groups(self):
        return self._groups

    @property
    def group_cmd_args(self):
        return self._group_cmd_args

    @property
    def group_const_args(self):
        return self._group_const_args

    @property
    def group_descriptions(self):
        return self._group_descriptions


def arg(*args, **kwargs):
    return _Args.CmdlineArg(*args, **kwargs)


def _init_parser(args):
    num_cmds = len(args.groups)
    if num_cmds == 0:
        raise RuntimeError("At least one command should be defined.")

    parser = ArgumentParser()

    for name, value in args.common_cmd_args.items():
        parser.add_argument(*value.args, **value.kwargs)

    if num_cmds == 1:
        """register arguments as common"""
        group = args.groups[0]
        for name, value in args.group_cmd_args[group].items():
            parser.add_argument(*value.args, **value.kwargs)
    else:
        """register arguments as groups"""
        subparsers = parser.add_subparsers(
            title='commands', help='available commands', dest='command')
        subparsers.required = True
        for group in args.groups:
            subparser = subparsers.add_parser(
                group, help=args.group_descriptions[group])
            for name, value in args.group_cmd_args[group].items():
                subparser.add_argument(*value.args, **value.kwargs)

    return parser


def _parse_args(def_args, command=None):
    parser = _init_parser(def_args)
    grouped = len(def_args.groups) > 1

    _args = def_args.sys_argv[1:]
    if command is not None:
        if command not in def_args.groups:
            raise ValueError("Undefined command is specified.")
        if grouped:
            _args = [command].extend(_args)

    parsed_args = vars(parser.parse_args(_args))
    if not grouped:
        parsed_args['command'] = def_args.groups[0]
    group = parsed_args['command']
    assert command is None or command == group
    command_args = {arg: parsed_args[arg]
                    for arg in def_args.group_cmd_args[group].keys()}
    common_args = {arg: parsed_args[arg]
                   for arg in def_args.common_cmd_args.keys()}
    for name, value in def_args.group_const_args[group].items():
        command_args[name] = value
    for name, value in def_args.common_const_args.items():
        common_args[name] = value
    command = group

    return command, command_args, common_args


def _read_config(file, sections, prefix=None):
    if prefix is None:
        prefix = ''

    config = {}
    parser = ConfigParser()
    parser.read(os.path.expanduser(file))

    for section in sections:
        section_name = prefix + '.' + section
        _config = {}
        if section_name in parser:
            for name, value in parser.items(section_name):
                if re.match(r"^(?:true|false)$", value, re.IGNORECASE):
                    value = parser[section_name].getboolean(name)
                elif re.match(r"^\d+$", value):
                    value = int(value)
                elif re.match(r"^\d+\.\d+$", value):
                    value = float(value)
                _config[name] = value
        config[section] = _config

    return config


class AppBase(Singleton):
    __instance = None
    _commands = {}
    _args = _Args()

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def add_command(cls, name, command, args={}, description=None):
        assert type(args) == dict
        cls._args.add_group(name, description)
        for arg_name, value in args.items():
            cls.add_arg(arg_name, value, group=name)
        cls._commands[name] = command

    @classmethod
    def add_arg(cls, name, value, group=None):
        if group is None:
            cls._args.def_arg(name, value)
        else:
            cls._args.def_group_arg(group, name, value)

    @classmethod
    def configure(cls, **kwargs):
        pass

    @classmethod
    def _parse_args(cls, command=None):
        return _parse_args(AppBase._args, command)

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
                      exc_info=True, stack_info=True)
        except SystemExit as e:
            logging.w(e)
        finally:
            logging.getLogger().finalize()
            sys.exit(0)

    @classmethod
    def _get_instance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            cls.__instance = instance
        return cls.__instance

    def _initialize(self, command, args, config):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._command = AppBase._commands[command]
        self._command_args = args
        self._config = config
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

    @classmethod
    def configure(cls, **kwargs):
        if hasattr(cls, '_configured') and cls._configured:
            return
        cls.app_name = (kwargs['name'] if 'name' in kwargs
                        else cls.DEFAULT_APP_NAME)
        script_name = cls._args.sys_argv[0]
        basedir = os.path.dirname(os.path.realpath(script_name))
        default_logdir = basedir + '/logs'
        loglevel = (kwargs['loglevel']
                    if 'loglevel' in kwargs else logging.INFO)
        cls.add_arg('basedir', basedir)
        cls.add_arg('debug', arg('--debug',
                                 type=str,
                                 default=False,
                                 help='Enable debug mode'))
        cls.add_arg('logdir', arg('--logdir',
                                  type=str,
                                  default=default_logdir,
                                  help='Log directory'))
        cls.add_arg('logoption', arg('--logoption',
                                     type=str,
                                     default='a',
                                     help='Log option: {a,d,h,n,w}'))
        cls.add_arg('loglevel', loglevel)
        cls.add_arg('script_name', script_name)
        cls.add_arg('quiet', arg('--quiet',
                                 action='store_true',
                                 default=False,
                                 help='execute quietly: '
                                 'does not print any message'))
        cls._configured = True

    @classmethod
    def _parse_args(cls, command=None):
        command, command_args, common_args \
            = super(cls, App)._parse_args(command)
        commands = list(cls._commands.keys())
        commands.append('common')
        config = _read_config(cls.DEFAULT_CONFIG_FILE,
                              commands, prefix=cls.app_name)
        common_args.update(config['common'])
        command_args.update(config[command])
        return command, command_args, common_args

    def _preprocess(self):
        uname = os.uname()
        verbose = not(self._config['quiet'])

        if self._config['debug']:
            if (self._config['loglevel'] < logging.DISABLE
                    and self._config['loglevel'] > logging.DEBUG):
                self._config['loglevel'] = logging.DEBUG
        logger_name = self._config['script_name']
        logger_config = {
            'logdir': self._config['logdir'],
            'level': self._config['loglevel'],
            'verbosity': logging.TRACE if verbose else logging.DISABLE
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
        if not verbose:
            sys.stdout = sys.stderr = open(os.devnull, 'w')

        logger.v(str(os.uname()))
        logger.v("sys.argv: %s" % str(sys.argv))
        logger.v("app._config: {}".format(self._config))
        logger.i("*** [START] ***")

    def _postprocess(self):
        logging.i("*** [DONE] ***")
