# from abc import abstractmethod
from argparse import ArgumentParser
import os
import signal
import sys

from ..base import Singleton
# from .. import logging


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
        subparsers = parser.add_subparsers(title='commands', help='available commands', dest='command')
        subparsers.required = True
        for group in args.groups:
            subparser = subparsers.add_parser(group, help=args.group_descriptions[group])
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
    command_args = {arg: parsed_args[arg] for arg in def_args.group_cmd_args[group].keys()}
    common_args = {arg: parsed_args[arg] for arg in def_args.common_cmd_args.keys()}
    for name, value in def_args.group_const_args[group].items():
        command_args[name] = value
    for name, value in def_args.common_const_args.items():
        common_args[name] = value
    command = group

    return command, command_args, common_args


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
    def run(cls, command=None):
        command, command_args, common_args = _parse_args(cls._args, command)
        """
        try {
            # // umaskは共通
            # umask(0);
            // インスタンスの生成
            $oSelf = static::_getInstance();
            // 初期化
            $oSelf->_initialize();
            // コントローラの処理実行
            $oSelf->_preProcess();
            $oSelf->_process();
            $oSelf->_postProcess();
            // 終了化
            $oSelf->_finalize();
        } catch (\Exception $oException) {
            // ログを出力
            \Log::exception($oException, __METHOD__ . '::' . __LINE__);
            // 例外画面の出力
            call_user_func(static::ERRORHANDLER, $oException, 500);
        }
        return true;
        """

        print(command_args, common_args)

    @classmethod
    def _get_instance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            instance._initialize()
            cls.__instance = instance
        return cls.__instance

    def _initialize(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._basedir = os.path.dirname(os.path.realpath(__file__))
        self._logdir = self._basedir + '/logs'
        # self._loglevel = Logger.DEBUG
        self._verbose = True
        self._debug = True

        self._initialized = True

    def _preprocess(self, *args, **kwargs):
        pass

    def _process(self):
        pass

    def _postprocess(self, *args, **kwargs):
        pass

    # def _exec(self, func, args):

    @classmethod
    def configure(cls):
        pass

    """

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
