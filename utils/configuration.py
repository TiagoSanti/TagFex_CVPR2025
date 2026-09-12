import yaml
from pathlib import Path

def load_yaml(p):
    with Path(p).open('r') as f:
        content = yaml.safe_load(f)
    return content

def load_configs(config_list):
    configs = dict()
    for config in config_list:
        configs.update(load_yaml(config))
    return configs


def training_config(args):
    """Apply the train CLI's shallow YAML precedence, including force_no_debug."""
    configs = vars(args)
    configs.update(load_configs(args.exp_configs))
    if args.force_no_debug:
        configs['debug'] = False
    return configs


def queue_training_config(config_list, seed):
    """Use the real train parser defaults, without importing or starting training."""
    from utils.argument import get_args
    args = get_args(['train', '--seed', str(seed), '--exp-configs', *map(str, config_list)])
    return training_config(args)
