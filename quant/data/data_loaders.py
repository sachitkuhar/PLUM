
"""Data loaders for MNIST, CIFAR-10, CIFAR-100, and ImageNet datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
import typing as t

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms



from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, ToTensor, ToDevice, ToTorchImage, Convert, NormalizeImage, Squeeze
import torchvision as tv
from ffcv.fields.basics import IntDecoder
import numpy as np
from MLclf import MLclf
import torch
import torchvision.transforms as transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


class QuantDataLoader(ABC):
    """Abstract class from which to instantiate training and test set PyTorch data loaders."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        world_size: int,
        download: bool = True,
        test_sampler: t.Optional[Sampler] = None,
    ):
        """
        Construct QuantDataLoader object, used for obtaining training and test set loaders.

        Args:
            train_batch_size: training set batch size
            test_batch_size: test set batch size
            dataset_path: root location of the dataset
            workers: number of workers to use for the data loader
            download: whether to download dataset.
                If false `dataset_path` should contain pre-downloaded dataset.
            test_sampler: PyTorch data sampler for the test set
        """
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_path = dataset_path
        self.workers = workers
        self.download = download
        self.test_sampler = test_sampler

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        raise NotImplementedError

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean up any temporary data."""
        pass


class ImageNetDataLoader(QuantDataLoader):
    """
    Subclass of :class:`~quant.data.data_loaders.QuantDataLoader`, for ImageNet.

    The dataset must already be available and cannot be downloaded by this data loader.
    """

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        dataset_path: str,
        workers: int,
        world_size: int,
        device: int,
        download: bool = False,
        test_sampler: t.Optional[Sampler] = None,
        train_split: str = "train",
        val_split: str = "val",
    ):
        """Construct a class for getting ImageNet data loaders."""
        super(ImageNetDataLoader, self).__init__(
            train_batch_size,
            test_batch_size,
            dataset_path,
            workers,
            world_size,
            download,
            test_sampler,
        )
        if download:
            raise ValueError(
                'ImageNet must be downloaded manually due to licensing restrictions.'
            )
        self.train_split = train_split
        self.val_split = val_split
        self.distributed = True #if world_size > 1 else False
        self.device = device
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.workers = workers

    def get_train_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the training set."""
        train_sampler = None

        res_tuple = (224, 224)
        cropper = RandomResizedCropRGBImageDecoder(res_tuple)
        image_pipeline = [
            cropper,
            RandomHorizontalFlip(),
            ToTensor(),
            # Move to GPU asynchronously as uint8:
            ToDevice(torch.device(self.device), non_blocking=True),
            # Automatically channels-last:
            ToTorchImage(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            # Standard torchvision transforms still work!
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
        ]
        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]

        order = OrderOption.RANDOM

        train_loader = '/home/sachitkuhar/unary_imagenet2/data/train_500_0.50_90.ffcv'

        train_loader = Loader(train_loader,
                        batch_size=self.train_batch_size,
                        num_workers=self.workers,
                        order=order,
                        os_cache=True,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=True)

        return train_loader, train_sampler

    def get_test_loader(self) -> DataLoader:
        """Get a PyTorch data loader for the test set."""

        res_tuple = (224, 224)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(self.device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]
        val_dataset='/home/sachitkuhar/unary_imagenet2/data/val_500_0.50_90.ffcv'
        loader = Loader(val_dataset,
                        batch_size=self.test_batch_size,
                        # num_workers=self.workers,
                        num_workers=1,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=True)

        return loader


