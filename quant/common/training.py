import logging
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer  # type: ignore
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader


from quant.common.metrics import Metric
import builtins
import sys

logger = logging.getLogger(__name__)


def _get_lr(optimizer: Optimizer) -> float:
    """
    Get learning rate of the first parameter group.

    Args:
        optimizer (optim.Optimizer): PyTorch optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

    raise ValueError('Cannot get optimizer LR: optimizer does not have any parameter groups.')


def project(optimizer: Optimizer) -> None:
    """Project model parameters to a range so that they can be updated."""
    # No-op
    # In theory, we should project the quantized weights to the [-1, 1] range
    # so that we have non-zero gradients and they can be updated.
    # However, in practice, we notice that this does not make a difference.
    # Hence, this is a no-op.
    _ = optimizer
    return None

def ede(model, optimizer, current_epoch, total_epochs):

    curr_lr = optimizer.param_groups[0]['lr']
    device = model.device

    T_min = torch.tensor(10**-1, device=model.device)
    T_max = torch.tensor(10**1, device=model.device)

    t = T_min * torch.pow(10, (current_epoch / total_epochs) * torch.log10(T_max / T_min))
    k = torch.max(1/t, torch.tensor(1., device=device))

    for n, p in model.named_parameters():
        if 'blocks' in n and 'conv' in n and 'weight' in n:
            p_data_device = p.data.to(device)
            p_grad_device = p.grad.to(device)
            delta = (p_data_device.abs().max(dim=0, keepdim=True)[0] * 0.05)
            delta = delta.expand_as(p_data_device)

            grad_input_first_half = k * t * (1 - torch.tanh((p_data_device[:p_data_device.shape[0]//2] - delta[:delta.shape[0]//2]) * t)**2) * p_grad_device[:p_grad_device.shape[0]//2]
            grad_input_second_half = k * t * (1 - torch.tanh((p_data_device[p_data_device.shape[0]//2:] + delta[delta.shape[0]//2:]) * t)**2) * p_grad_device[p_grad_device.shape[0]//2:]

            grad_input = torch.cat([grad_input_first_half, grad_input_second_half])

            p.grad = grad_input

def train(
    model: Union[nn.Module, nn.DataParallel],
    train_loader: DataLoader,
    metrics: Dict[str, Metric],
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int,
    teacher: Optional[Union[nn.Module, nn.DataParallel]] = None,
) -> Dict[str, float]:
    """
    Train a model on some data using some criterion and with some optimizer.

    Args:
        model: Model to train
        train_loader: Data loader for loading training data
        metrics: A dict mapping evaluation metric names to metrics classes
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler
        device: PyTorch device object
        epoch: Current epoch, where the first epoch should start at 1
        log_interval: Number of batches before printing loss
        hooks: A sequence of functions that can implement custom behavior
        teacher: teacher network for knowledge distillation, if any

    Returns:
        A dictionary mapping evaluation metric names to computed values for the training set.
    """
    print("training stared")

    model.train()
    for metric in metrics.values():
        metric.reset()

    loss_fn = model.module.loss_fn if isinstance(model, nn.parallel.DistributedDataParallel) else model.loss_fn

    seen_examples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        if teacher is None:
            teacher_output = None
            loss = loss_fn(output, target)  # type: ignore
        else:
            teacher_output = teacher(data)
            loss = loss_fn(output, teacher_output, target)  # type: ignore
        loss.backward()
        # ede(model, optimizer, epoch, 320)
        optimizer.step()
        project(optimizer)
        scheduler.step()  # type: ignore

        with torch.no_grad():
            for metric in metrics.values():
                metric.update(output, target, teacher_output=teacher_output)

        seen_examples += len(data)
        if batch_idx % log_interval == 0:
            logger.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                    epoch,
                    seen_examples,
                    len(train_loader),
                    100 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        break

    # Computing evaluation metrics for training set
    computed_metrics = {name: metric.compute() for name, metric in metrics.items()}

    logger.info('Training set evaluation metrics:')
    for name, metric in metrics.items():
        logger.info(f'{name}: {metric}')

    return computed_metrics

def evaluate(
    model: Union[nn.Module, nn.DataParallel],
    test_loader: DataLoader,
    metrics: Dict[str, Metric],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Evaluate model on some held-out set.

    Args:
        model: Model to test on
        test_loader: Data loader for loading test data
        metrics: A dict mapping evaluation metric names to metrics classes
        device: PyTorch device object
        epoch: Current epoch, where the first epoch should start at 1
        hooks: A sequence of functions that can implement custom behavior

    Returns:
        A dictionary mapping evaluation metric names to computed values.
    """

    model.eval()
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            output = model(data)

            for metric in metrics.values():
                metric.update(output, target)
            break


    computed_metrics = {name: metric.compute() for name, metric in metrics.items()}

    logger.info('Test set evaluation metrics:')
    for name, metric in metrics.items():
        logger.info(f'{name}: {metric}')

    return computed_metrics
