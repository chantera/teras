import argparse
from collections import OrderedDict
from configparser import ConfigParser
import os
import re
import sys


class CmdlineArg(object):

    def __init__(self, *args, **kwargs):
        assert len(args) > 0
        self._names = [name.lstrip('-') for name
                       in sorted(args, key=len, reverse=True)]
        self._args = args
        self._kwargs = kwargs

    @property
    def names(self):
        return self._names

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def add_arg(self, value):
        self._args.append(value)
        self._names.append(value.lstrip('-'))
        self._names.sort(key=len, reverse=True)

    def add_kwarg(self, name, value):
        self._kwargs[name] = value


def arg(*args, **kwargs):
    return CmdlineArg(*args, **kwargs)


class ArgDefinition(object):

    def __init__(self):
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
            self.def_group(group)
        if type(value) == CmdlineArg:
            value.add_kwarg('dest', name)
            self._group_cmd_args[group][name] = value
        else:
            self._group_const_args[group][name] = value

    def def_group(self, group, **kwargs):
        if group not in self._groups:
            self._groups.append(group)
            self._group_cmd_args[group] = {}
            self._group_const_args[group] = {}
            self._group_descriptions[group] = kwargs
        elif kwargs:
            self._group_descriptions[group] = kwargs

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
    def grouped_cmd_args(self):
        return self._group_cmd_args

    @property
    def grouped_const_args(self):
        return self._group_const_args

    @property
    def group_descriptions(self):
        return self._group_descriptions


class ArgParser(object):
    DEFAULT_FORMATTER_CLASS = argparse.ArgumentDefaultsHelpFormatter

    def __init__(self):
        self._def = ArgDefinition()

    def add_arg(self, name, value, group=None):
        if group is None:
            self._def.def_arg(name, value)
        else:
            self._def.def_group_arg(group, name, value)

    def add_group(self, group, **kwargs):
        self._def.def_group(group, **kwargs)

    def parse(self, args=None, parser=None, command=None):
        if args is None:
            args = sys.argv[1:]
        if parser is None:
            parser = self._init_parser(
                formatter_class=ArgParser.DEFAULT_FORMATTER_CLASS)
        grouped = len(self._def.groups) > 1

        if command is not None:
            if command not in self._args.groups:
                raise ValueError("Undefined command is specified.")
            if grouped:
                args = [command].extend(args)  # specify command

        _def = self._def
        parsed_args = vars(parser.parse_args(args))
        if not grouped:
            parsed_args['command'] = _def.groups[0]
        group = parsed_args['command']
        assert command is None or command == group
        command_args = {arg: parsed_args[arg]
                        for arg in _def.grouped_cmd_args[group].keys()}
        common_args = {arg: parsed_args[arg]
                       for arg in _def.common_cmd_args.keys()}
        for name, value in _def.grouped_const_args[group].items():
            command_args[name] = value
        for name, value in _def.common_const_args.items():
            common_args[name] = value
        command = group

        return command, command_args, common_args

    def _init_parser(self, **kwargs):
        _def = self._def
        num_groups = len(_def.groups)
        if num_groups == 0:
            raise RuntimeError("At least one command should be defined.")

        formatter_class = argparse.HelpFormatter
        if 'formatter_class' in kwargs:
            formatter_class = kwargs['formatter_class']
        parser = argparse.ArgumentParser(**kwargs)

        for name, value in _def.common_cmd_args.items():
            parser.add_argument(*value.args, **value.kwargs)

        if num_groups == 1:
            """register arguments as common"""
            group = _def.groups[0]
            for name, value in _def.grouped_cmd_args[group].items():
                parser.add_argument(*value.args, **value.kwargs)
        else:
            """register arguments as groups"""
            subparsers = parser.add_subparsers(
                title='commands', help='available commands', dest='command')
            subparsers.required = True
            for group in _def.groups:
                subparser = subparsers.add_parser(
                    group, **_def.group_descriptions[group],
                    formatter_class=formatter_class)
                for name, value in _def.grouped_cmd_args[group].items():
                    subparser.add_argument(*value.args, **value.kwargs)

        return parser


class ConfigArgParser(ArgParser):
    _DEBUG = False

    def __init__(self, default_config_file):
        super(ConfigArgParser, self).__init__()
        self._default_config_file = default_config_file
        self._cmd_args_map = {}
        self._config = None
        self._source = None

    def add_arg(self, name, value, group=None):
        if type(value) is CmdlineArg:
            cmd_arg_name = ((group if group is not None else 'common'),
                            value.names[0])
            self._cmd_args_map[cmd_arg_name] = name
        return super(ConfigArgParser, self).add_arg(name, value, group)

    def parse(self, args=None, parser=None, command=None, section_prefix=None):
        if args is None:
            args = sys.argv[1:]
        if parser is not None:
            return super(ConfigArgParser, self).parse(args, parser, command)

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config',
                            type=str,
                            default=self._default_config_file,
                            help='configuration file',
                            metavar='FILE')
        parser.add_argument('--saveconfig',
                            type=str,
                            help='save current configuration to file',
                            metavar='FILE')
        namespace, args = parser.parse_known_args(args)
        config_file = os.path.expanduser(namespace.config)

        if os.path.exists(config_file):
            groups = list(self._def.groups)
            groups.append('common')
            self._config = self._read_config(config_file, groups,
                                             section_prefix)
            self._source = config_file
        elif config_file != os.path.expanduser(self._default_config_file):
            raise FileNotFoundError("config file was not found: "
                                    "'%s'" % config_file)

        parent_parser = parser
        parser = self._init_parser(
            parents=[parent_parser],
            formatter_class=ConfigArgParser.DEFAULT_FORMATTER_CLASS)

        command, command_args, common_args = \
            super(ConfigArgParser, self).parse(args, parser, command)
        if namespace.saveconfig is not None:
            new_config_file = os.path.expanduser(namespace.saveconfig)
            current_config = OrderedDict()
            current_config['common'] = common_args
            current_config[command] = command_args
            self._write_config(new_config_file, current_config, section_prefix)
        return command, command_args, common_args

    def _read_config(self, file, sections, prefix=None,
                     ignore_undefined_args=True):
        if prefix is None:
            prefix = ''

        config = {}
        parser = ConfigParser()
        parser.read(os.path.expanduser(file))
        if ConfigArgParser._DEBUG:
            print("config file loaded: {}".format(file), file=sys.stderr)

        for section in sections:
            config[section] = self._read_config_section(
                parser, section, prefix, ignore_undefined_args)

        return config

    def _read_config_section(self, config_parser, section, prefix,
                             ignore_undefined_args=True):
        _config = {}
        section_name = prefix + '.' + section
        if section_name not in config_parser:
            return _config

        def_args = (self._def.common_cmd_args if section == 'common'
                    else self._def.grouped_cmd_args[section])
        for name, value in config_parser.items(section_name):
            cmd_arg_name = (section, name)
            if cmd_arg_name in self._cmd_args_map:
                dist_name = self._cmd_args_map[cmd_arg_name]
            else:
                if not ignore_undefined_args:
                    _config[name] = value
                continue

            def_arg = def_args[dist_name]
            value = self._cast_value(value, def_arg)

            def_arg.kwargs['default'] = value
            if 'required' in def_arg.kwargs:
                del def_arg.kwargs['required']

            _config[dist_name] = value
        return _config

    def _write_config(self, file, config, prefix=None):
        if prefix is None:
            prefix = ''

        parser = ConfigParser()
        file = os.path.expanduser(file)
        if ConfigArgParser._DEBUG or True:
            print("write config to file: {}".format(file), file=sys.stderr)

        for section, _config in config.items():
            section_name = prefix + '.' + section
            def_args = (self._def.common_cmd_args if section == 'common'
                        else self._def.grouped_cmd_args[section])
            new_config = {}
            for name, value in _config.items():
                if name in def_args:
                    new_config[def_args[name].names[0]] = value
            parser[section_name] = new_config

        with open(file, 'w') as configfile:
            parser.write(configfile)

    @staticmethod
    def _cast_value(value, def_arg):
        if 'type' in def_arg.kwargs:
            _type = def_arg.kwargs['type']
            value = _type(value) if _type is not bool else _getboolean(value)
        elif 'default' in def_arg.kwargs:
            _type = type(def_arg.kwargs['default'])
            value = _type(value) if _type is not bool else _getboolean(value)
        elif re.match(r"^(?:true|false)$", value, re.IGNORECASE):
            value = _getboolean(value)
        elif re.match(r"^\d+$", value):
            value = int(value)
        elif re.match(r"^\d+\.\d+$", value):
            value = float(value)
        return value

    @property
    def config(self):
        return self._config

    @property
    def source(self):
        return self._source


_BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True,
                   '0': False, 'no': False, 'false': False, 'off': False}


def _getboolean(value):
    return _BOOLEAN_STATES[value.lower()]
