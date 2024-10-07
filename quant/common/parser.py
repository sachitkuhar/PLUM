from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable
import yaml

import torch


def _validate_args(args: Namespace) -> None:
    """
    Validate arguments.

    Args:
        args:  parsed argparse CLI args
    """
    if not args.restore_experiment and not args.config:
        raise ValueError('--config must be specified if not restoring from experiment.')

    if args.restore_experiment and args.init_from_checkpoint:
        raise ValueError('Only one of --restore-experiment / --init-from-checkpoint can be set.')


def parse_common_fields(args: Namespace, config: dict) -> None:
    """
    Populate common fields in the config with parsed args.

    Args:
        args: parsed argparse CLI args
        config: config dictionary storing final resolved args
    """
    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
    else:
        from datetime import datetime

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        config_name_without_ext = Path(config['config']).stem
        config['experiment_name'] = f'{current_time}_{config_name_without_ext}'

    if 'environment' not in config or 'platform' not in config['environment']:
        config['environment'] = {'platform': 'local'}

    if args.ngpus is not None:
        config['environment']['ngpus'] = args.ngpus
    if 'ngpus' not in config['environment']:
        config['environment']['ngpus'] = 1 if torch.cuda.is_available() else 0

    config['skip_training'] = args.skip_training

    if args.init_from_checkpoint:
        config['init_from_checkpoint'] = args.init_from_checkpoint


def parse_config(args: Namespace, validator: Callable[[Namespace], None] = _validate_args) -> dict:
    """
    Parse config file and override with CLI args.

    Args:
        args: parsed argparse CLI args
        validator: validator for config

    Returns:
        A resolved config, applying CLI args on top of the config file
    """
    validator(args)

    config = {}
    if args.restore_experiment:
        with open(Path(args.restore_experiment) / 'config.yaml') as f:
            config = yaml.safe_load(f)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config['config'] = args.config

    parse_common_fields(args, config)

    if args.restore_experiment_epoch:
        config['restore_experiment_epoch'] = int(args.restore_experiment_epoch)

    if args.restore_experiment:
        config['restore_experiment'] = args.restore_experiment

    return config


def get_base_argument_parser(description: str) -> ArgumentParser:
    """
    Get a base argument parser for driver scripts.

    Args:
        description: A string describing the driver script.

    Returns:
        Parser object to extend.
    """
    parser = ArgumentParser(description)
    parser.add_argument('--config', type=str, help='Path to a yaml config file.')
    parser.add_argument(
        '--experiment-name', type=str, default=None, help='Name of the experiment.'
    )
    parser.add_argument(
        '--ngpus', type=int, default=None, help='Number of GPUs. Use 0 for CPU.'
    )
    parser.add_argument(
        '--skip-training',
        default=False,
        action='store_true',
        help='Skip training and only run evaluation. Checkpoint must be passed in as well.',
    )
    parser.add_argument(
        '--restore-experiment',
        type=str,
        help='Path to experiments directory to restore checkpoint from.',
    )
    parser.add_argument(
        '--init-from-checkpoint',
        type=str,
        help='Path to model file to initialize model parameters.',
    )
    parser.add_argument(
        '--restore-experiment-epoch',
        type=int,
        help='restore experiment epoch.',
    )
    return parser
