# from abc import abstractmethod
from argparse import ArgumentParser
import os
import signal
import sys

from ..base import Singleton
# from .. import logging


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


class _CommandArgs(object):

    def __init__(self):
        self._sys_argv = sys.argv
        self._common_cmd_args = {}
        self._common_const_args = {}
        self._groups = []
        self._group_cmd_args = {}
        self._group_const_args = {}
        self._group_descriptions = {}

    def def_arg(self, name, value):
        if type(value) == CmdlineArg:
            value.add_kwarg('dest', name)
            self._common_cmd_args[name] = value
        else:
            self._common_const_args[name] = value

    def def_group_arg(self, group, name, value):
        if group not in self._groups:
            self.add_group(group)
        if type(value) == CmdlineArg:
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

    def _init_parser(self):
        num_cmds = len(self._groups)
        if num_cmds == 0:
            raise RuntimeError("At least one command should be defined.")

        parser = ArgumentParser()

        for name, value in self._common_cmd_args.items():
            parser.add_argument(*value.args, **value.kwargs)

        if num_cmds == 1:
            """register arguments as common"""
            group = self._groups[0]
            for name, value in self._group_cmd_args[group].items():
                parser.add_argument(*value.args, **value.kwargs)
        else:
            """register arguments as groups"""
            subparsers = parser.add_subparsers(title='commands', help='available commands', dest='command')
            subparsers.required = True
            for group in self._groups:
                subparser = subparsers.add_parser(group, help=self._group_descriptions[group])
                for name, value in self._group_cmd_args[group].items():
                    subparser.add_argument(*value.args, **value.kwargs)

        return parser

    def parse(self, command=None):
        parser = self._init_parser()
        grouped = len(self._groups) > 1

        _args = self._sys_argv[1:]
        if command is not None:
            if command not in self._groups:
                raise ValueError("Undefined command is specified.")
            if grouped:
                _args = [command].extend(_args)

        args = vars(parser.parse_args(_args))
        if not grouped:
            args['command'] = self._groups[0]
        group = args['command']
        assert command is None or command == group
        command_args = {arg: args[arg] for arg in self._group_cmd_args[group].keys()}
        common_args = {arg: args[arg] for arg in self._common_cmd_args.keys()}
        for name, value in self._group_const_args[group].items():
            command_args[name] = value
        for name, value in self._common_const_args.items():
            common_args[name] = value

        return command_args, common_args


def arg(*args, **kwargs):
    return CmdlineArg(*args, **kwargs)


class AppBase(Singleton):
    __instance = None

    def __init__(self):
        raise NotImplementedError()

    def _initialize(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._basedir = os.path.dirname(os.path.realpath(__file__))
        self._logdir = self._basedir + '/logs'
        # self._loglevel = Logger.DEBUG
        self._verbose = True
        self._debug = True

        self._commands = {}
        self._args = _CommandArgs()

        self._initialized = True

    @classmethod
    def _get_instance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            instance._initialize()
            cls.__instance = instance
        return cls.__instance

    @classmethod
    def add_command(cls, name, command, args={}, description=None):
        assert type(args) == dict
        instance = cls._get_instance()
        instance._args.add_group(name, description)
        for arg_name, value in args.items():
            instance._add_arg(arg_name, value, group=name)
        instance._commands[name] = command

    @classmethod
    def add_arg(cls, name, value, group=None):
        return cls._get_instance()._add_arg(name, value, group)

    def _add_arg(self, name, value, group=None):
        if group is None:
            self._args.def_arg(name, value)
        else:
            self._args.def_group_arg(group, name, value)

    @classmethod
    def run(cls, command=None):
        command_args, common_args = cls._get_instance()._args.parse(command)
        print(command_args, common_args)

    """
    def _parse_args(self):
        num_cmds = len(self._commands)
        if num_cmds == 0:
            raise RuntimeError("At least one command should be defined.")
        elif num_cmds == 1:
            pass

        constants = {}

        parser = ArgumentParser()
        _constants = {}
        for arg, value in self._defined_args[AppBase.DEFAULT_ARG_GROUP].items():
            if type(value) == CmdArg:
                kwargs = value.kwargs
                kwargs['dest'] = arg
                parser.add_argument(*value.args, **kwargs)
            else:
                _constants[arg] = value
        constants[AppBase.DEFAULT_ARG_GROUP] = _constants

        subparsers = parser.add_subparsers(title='commands', dest='command')
        subparsers.required = True
        for command in self._commands.keys():
            _constants = {}
            subparser = subparsers.add_parser(command)
            for arg, value in self._defined_args[command].items():
                if type(value) == CmdArg:
                    kwargs = value.kwargs
                    kwargs['dest'] = arg
                    subparser.add_argument(*value.args, **kwargs)
                else:
                    _constants[arg] = value
            constants[command] = _constants

        args = vars(parser.parse_args())
        for group_name in (AppBase.DEFAULT_ARG_GROUP, args['command']):
            print(type(args), args)
            print(constants[group_name])
            args.update(constants[group_name])
            # for arg, value in constants[group_name]:
            #     if arg not in args:
            #         args[arg] = value
            #     if not hasattr(self, arg):
            #         setattr(self, arg, value)

        return args
    """

    @classmethod
    def configure(cls):
        pass

    # @classmethod
    # def configure(
    #         self,
    #         logdir=_logdir,
    #         loglevel=_loglevel,
    #         verbose=_verbose,
    #         debug=_debug):
    #     self._loglevel = loglevel
    #     self._verbose = verbose
    #     self._logdir = logdir
    #     self._debug = debug

    @classmethod
    def _def_arg(cls, *args, **kwargs):
        cls.__defined_args.append((args, kwargs))

    """
    def __init__(self):
        self._basedir = os.path.dirname(os.path.realpath(getfile(self.__class__)))
        self._logdir = self._basedir + '/logs'

    def __initialize(self):
        self._initialize()
        self._name = sys.argv[0]
        self._def_arg('--debug', type=str, default=self._debug,
                      help='Enable debug mode')
        self._def_arg('--logdir', type=str, default=self._logdir,
                      help='Log directory')
        self._def_arg('--silent', '--quiet', action='store_true', default=not(self._verbose),
                      help='Silent execution: does not print any message')
        parser = argparse.ArgumentParser()
        [parser.add_argument(*_args, **_kwargs) for (_args, _kwargs) in self.__defined_args]
        args = parser.parse_args()
        self.configure(args.logdir, Logger.DEBUG if args.debug else Logger.INFO, not(args.silent), args.debug)
        Logger.configure(loglevel=self._loglevel, verbose=self._verbose, logdir=self._logdir)
        if not self._verbose:
            sys.stdout = sys.stderr = open(os.devnull, 'w')
        self._args = args

    def __call__(self):
        self.__initialize()
        try:
            def handler(signum, frame):
                raise SystemExit("Signal(%d) received: The program %s will be closed" % (signum, __file__))
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
            self.main()
        except Exception:
            Logger.e("Exception occurred during execution:")
        except SystemExit as e:
            Logger.w(e)
        finally:
            Logger.finalize()
            sys.exit(0)
    """


class App(AppBase):
    pass
