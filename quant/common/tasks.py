"""Utilities for running tasks."""
import builtins
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import yaml

import torch
import torch.nn as nn

from quant import Hook, MetricDict
from quant.common import init_logging
from quant.utils.checkpoints import get_path_to_checkpoint, log_checkpoints, \
    restore_from_checkpoint
from quant.common.initialization import (
    get_device,
    get_model,
    get_optimizer,
    get_lr_scheduler,
    get_loss_fn,
)
from quant.common.metrics import LossMetric, Top1Accuracy, TopKAccuracy
from quant.common.training import evaluate, train
from quant.data.data_loaders import QuantDataLoader
from quant.utils.kd_criterion import kd_criterion

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pandas as pd

def log_metrics_to_experiments_dir(
    train_epoch_metrics: List[dict],
    test_epoch_metrics: List[dict],
    experiment_root_directory: Path,
    experiment_name: str,
    skip_training: bool = False,
    gpu: int = 0,
) -> None:
    """
    Log metrics to experiments directory.

    Args:
        train_epoch_metrics: List of training metrics for every epoch
        test_epoch_metrics: List of test metrics for every epoch
        experiment_root_directory: root directory for storing logs, checkpoints, etc.
        experiment_name: Name of experiment
        skip_training: whether to log only eval metrics
    """
    metrics_dir = experiment_root_directory / experiment_name / 'metrics'
    metrics_dir.mkdir(exist_ok=True, parents=True)

    if not skip_training:
        train_metrics_df = pd.DataFrame.from_records(train_epoch_metrics)
        train_metrics_df.to_csv(str(metrics_dir)+ str('/')+ 'train' + str(gpu) +'.csv', index=False)

    test_metrics_df = pd.DataFrame.from_records(test_epoch_metrics)
    test_metrics_df.to_csv(str(metrics_dir)+ str('/')+  'test' + str(gpu) +'.csv', index=False)

def get_teacher_and_kd_loss(
    gpu: int,
    teacher_config_path: str,
    teacher_checkpoint_path: str,
    train_mode: bool,
    criterion_config: dict,
    device: torch.device,
    ngpus: int,
    freeze_teacher: bool = True,
    strict_keys: bool = True,
) -> Tuple[Union[nn.Module, nn.DataParallel], Callable[..., torch.Tensor]]:
    """
    Get teacher and KD loss for knowledge distillation.

    Args:
        teacher_config_path: path to config used to train teacher
        teacher_checkpoint_path: path to checkpoint to use to initialize teacher
        train_mode: if true, use teacher in train mode, or use eval mode otherwise
        criterion_config: config for KD criterion, such as alpha and temperature
        device: PyTorch device used to store teacher, should the be the same as model
        ngpus: number of GPUs to run teacher, should be the same as that of the student model
        freeze_teacher: whether to freeze teacher
        strict_keys: whether to enforce keys must exactly match for restoring checkpoint

    Returns:
        An initialized teacher and KD loss function with teacher-related args resolved
    """
    with open(teacher_config_path) as f:
        teacher_config = yaml.safe_load(f)
        teacher_model_config = teacher_config['model']

    loss_fn = get_loss_fn(teacher_model_config['loss'])
    teacher = get_model(
        gpu=gpu,
        architecture=teacher_model_config['architecture'],
        loss_fn=loss_fn,
        arch_config=teacher_model_config['arch_config'],
        device=device,
        ngpus=ngpus,
    )

    print("loading teacher now.")

    model, _, _, _ = restore_from_checkpoint(teacher, None, None, teacher_checkpoint_path, device, strict_keys)


    print("teacher loaded.2", flush=True)
    if freeze_teacher:
        print("freeze teacher", flush=True)
        for p in teacher.parameters():
            p.requires_grad_(False)
    else:
        print("dontfreeze teacher", flush=True)

    teacher.train() if train_mode else teacher.eval()

    kd_loss = partial(kd_criterion, freeze_teacher=freeze_teacher, **criterion_config)

    return teacher, kd_loss


def classification_task(
    gpu: int,
    config: dict,
    experiment_root_directory: Path,
    data_loader_cls: Type[QuantDataLoader],
    get_hooks: Callable[[dict, Path, MetricDict, MetricDict], Tuple[List[Hook], List[Hook]]],
    restore_experiment: Optional[Path] = None,
) -> None:#tuple[None, None]:
    """
    Driver program for running classification task.

    Args:
        config: merged config with CLI args
        experiment_root_directory: root directory for storing logs, checkpoints, etc.
        data_loader_cls: The QuantDataLoader class
        get_hooks: a function that returns lists of training and testing hooks
        restore_experiment: path to experiment to restore, None for do not restore

    Returns:
        (List of training set metrics for each epoch, list of test set metrics for each epoch).
    """
    env_config = config['environment']
    data_config = config['data']
    model_config = config['model']
    optimization_config = config['optimization']
    log_config = config['log']
    name = config['experiment_name']

    world_size = env_config['world_size'] # torch.cuda.device_count()

    if env_config['distributed'] and gpu!=0:
        print_ = lambda *args, **kwdargs: None
        builtins.print = print_
    if gpu is not None:
        builtins.print("Use GPU: {} for training".format(gpu))
    if env_config['distributed']:
        dist.init_process_group(backend='nccl', init_method=env_config["dist_url"],
                                world_size=world_size, rank=gpu)

    if gpu==0:
        init_logging(log_config['level'])

    if gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')

    epochs = optimization_config['epochs']

    cudnn.deterministic = env_config['cuda']['cudnn_deterministic']
    cudnn.benchmark = env_config['cuda']['cudnn_benchmark']

    if env_config['distributed']:
        if gpu is not None:
            torch.cuda.set_device(gpu)
            batch_size = config['data']['train_batch_size']
            workers = int((config['data']['workers'] + world_size - 1) / world_size)
        else:
            pass
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        batch_size = int(config['data']['train_batch_size'] / 1)
        workers = int((config['data']['workers'] + 1 - 1) / 1)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    teacher = None
    use_kd = 'kd_config' in model_config

    if use_kd:
        teacher, kd_loss = get_teacher_and_kd_loss(gpu=gpu,
            device=device, ngpus=env_config['ngpus'],
            strict_keys=model_config.get('strict_keys', True),
            **model_config['kd_config']
        )


    loss_fn = get_loss_fn(model_config['loss']) if not use_kd else kd_loss
    model = get_model(
        gpu=gpu,
        architecture=model_config['architecture'],
        loss_fn=loss_fn,
        arch_config=model_config['arch_config'],
        device=device,
        ngpus=env_config['ngpus'],
    )

    print("going for data loading")
    data_loader = data_loader_cls(dataset_path=data_config['dataset_path'], train_batch_size=batch_size, test_batch_size=data_config['test_batch_size'], workers=workers, world_size=world_size, device=device)
    if not config.get('skip_training'):
        train_loader, train_sampler = data_loader.get_train_loader()
    else:
        train_loader, train_sampler = None, None
    test_loader = data_loader.get_test_loader()
    print("data loaded")

    if train_loader is not None:
        print("len(train_loader)", len(train_loader))
    optimizer, scheduler = None, None
    if not config.get('skip_training'):
        optimizer = get_optimizer(model.parameters(), optimization_config['optimizer'])
        scheduler = get_lr_scheduler(optimizer, optimization_config['lr_scheduler'], epochs, len(train_loader))  # type: ignore  # noqa: E501

    if restore_experiment is not None:
        checkpoint_path = get_path_to_checkpoint(restore_experiment, config['restore_experiment_epoch'])
        model, restored_optimizer, restored_scheduler, start_epoch = restore_from_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_path,
            device,
            model_config.get('strict_keys', True),
        )
        optimizer, scheduler = restored_optimizer, restored_scheduler
        start_epoch += 1
    elif config.get('init_from_checkpoint'):
        model, _, _, _ = restore_from_checkpoint(
            model,
            None,
            None,
            config['init_from_checkpoint'],
            device,
            model_config.get('strict_keys', True),
        )
        start_epoch = 1
    else:
        start_epoch = 1

    train_metrics = {
        'Loss': LossMetric(loss_fn, accumulate=True),
        'Top-1 Accuracy': Top1Accuracy(accumulate=True),
        'Top-5 Accuracy': TopKAccuracy(5, accumulate=True),
    }

    test_metrics = {
        'Loss': LossMetric(get_loss_fn(model_config['loss']), accumulate=True),
        'Top-1 Accuracy': Top1Accuracy(accumulate=True),
        'Top-5 Accuracy': TopKAccuracy(5, accumulate=True),
    }

    train_hooks, test_hooks = None,None
    if device==torch.device('cuda:0'):
        train_hooks, test_hooks = get_hooks(config, experiment_root_directory,
                                            train_metrics, test_metrics)
    train_epoch_metrics, test_epoch_metrics = [], []

    if config.get('skip_training'):
        computed_test_metrics = evaluate(
            model=model,
            test_loader=test_loader,
            metrics=test_metrics,
            device=device,
            epoch=1,
        )
        test_epoch_metrics.append(computed_test_metrics)
    else:
        for epoch in range(start_epoch, epochs):
            if env_config['distributed']:
                pass
            print("start training")
            computed_train_metrics = train(
                model=model,
                train_loader=train_loader,  # type: ignore
                metrics=train_metrics,
                optimizer=optimizer,
                scheduler=scheduler,  # type: ignore
                device=device,
                epoch=epoch,
                log_interval=log_config['interval'],
                teacher=teacher,
            )
            computed_test_metrics = evaluate(
                model=model,
                test_loader=test_loader,
                metrics=test_metrics,
                device=device,
                epoch=epoch,
            )

            train_epoch_metrics.append(computed_train_metrics)
            test_epoch_metrics.append(computed_test_metrics)

            if device == torch.device('cuda:0'):
                if ((epoch <= 200 and epoch % 5 == 0) or 
                (200 < epoch <= 280 and epoch % 5 == 0) or
                (280 < epoch <= epochs)):
                    log_checkpoints(
                        experiment_root_directory / config['experiment_name'] / 'checkpoints',
                        model,
                        optimizer,  # type: ignore
                        scheduler,  # type: ignore
                        epoch,
            )
            break

    data_loader.cleanup()

    log_metrics_to_experiments_dir(
        train_epoch_metrics,
        test_epoch_metrics,
        experiment_root_directory,
        name,
        config['skip_training'],
        gpu
    )

    if env_config['distributed']:
        dist.destroy_process_group()

    return None
