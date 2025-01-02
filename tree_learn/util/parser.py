import argparse
import yaml
import os.path as osp
from munch import Munch

def get_args(args):
    parser = argparse.ArgumentParser('tree_learn')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--dist', action='store_true', help='distributed training')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


def load_yaml_file(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def get_config(config_path):
    # Get the main configuration file
    main_cfg = load_yaml_file(config_path)

    # Get the default arguments, which are paths to other YAML files
    default_args = main_cfg.pop('default_args', None)

    if default_args is not None:
        # Load the configuration from the default argument files
        for path in default_args:
            default_config = load_yaml_file(path)
            
            # Modify content of default args if specified so in main configuration and then update main configuration with modified default configuration
            for key in main_cfg:
                if key in default_config:
                    modify_default_cfg(default_config[key], main_cfg[key])
            
            main_cfg.update(default_config)
    return Munch.fromDict(main_cfg)


def get_args_and_cfg(args=None):
    args = get_args(args)
    cfg = get_config(args.config)
    print(args)
    if args.work_dir is not None:
        cfg.work_dir = osp.join('./work_dirs', args.work_dir)
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    return args, cfg


def modify_default_cfg(default_config, main_cfg):
    for key, value in main_cfg.items():
        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
            modify_default_cfg(default_config[key], value)
        else:
            default_config[key] = value

def munch_to_dict(obj):
    if isinstance(obj, Munch):
        return {key: munch_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [munch_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(munch_to_dict(item) for item in obj)
    else:
        return obj
