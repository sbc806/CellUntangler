# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from .vae_dataset import VaeDataset
from ..mvae.distributions import EuclideanUniform


# class UmiDataset:
#
#     def __init__(self, batch_size: int, in_dim: int) -> None:
#         self.batch_size = batch_size
#         self._in_dim = in_dim
#
#     def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError
#
#     def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         raise NotImplementedError
#
#     @property
#     def in_dim(self) -> int:
#         return self._in_dim
#
#     def metrics(self, x_mb_: torch.Tensor, mode: str = "train") -> Dict[str, float]:
#         return {}
#
#
# class ToDefaultTensor(transforms.Lambda):
#
#     def __init__(self) -> None:
#         super().__init__(lambda x: x.to(torch.get_default_dtype()))
#
#
# def flatten_transform(img: torch.Tensor) -> torch.Tensor:
#     return img.view(-1)
#
#
# class MnistVaeDataset(UmiDataset):
#
#     def __init__(self, batch_size: int, data_folder: str, in_dim: int) -> None:
#         super().__init__(batch_size, in_dim)
#         self.data_folder = data_folder
#
#     def _get_dataset(self, train: bool, transform: Any) -> torch.utils.data.Dataset:
#         return datasets.MNIST(self.data_folder, train=train, download=False, transform=transform)
#
#     def _load_mnist(self, train: bool) -> DataLoader:
#         transformation = transforms.Compose(
#             [transforms.ToTensor(),
#              ToDefaultTensor(),
#              transforms.Lambda(flatten_transform),
#              ])
#
#         return DataLoader(dataset=self._get_dataset(train, transform=transformation),
#                           batch_size=self.batch_size,
#                           num_workers=8,
#                           pin_memory=True,
#                           shuffle=train)
#
#     def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
#         train_loader = self._load_mnist(train=True)
#         test_loader = self._load_mnist(train=False)
#         return train_loader, test_loader
#
#     def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
#         return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")
#
# import anndata
# from torch.utils.data import DataLoader, Dataset
# from typing import Dict, Optional, Tuple


class UmiDataset(Dataset):
    def __init__(self, x, y=None, transforms=None):
        if y is not None:
            assert x.shape[0] == y.shape[0], \
                ('x.shape: %s, y.shape: %s' % (x.shape, y.shape))

        self.x = x
        self.y = y

        self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        data = self.x[i, :]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return torch.tensor(data), torch.tensor(self.y[i])
        else:
            return torch.tensor(data)


# adata = anndata.read_loom('/Users/jding/work/desc/hela_select.loom')
#
# x = adata.X.todense().astype(int)
# y = adata.obs['batch'].rank(method='dense').astype(int)
#
# dd = UmiDataset(x, y)
# dataloder = DataLoader(dataset=dd, batch_size=128)
#
# for i, data in enumerate(dataloder):
#     print(i)
#
#
# from mt.data.vae_dataset import VaeDataset
# from typing import Tuple, Any


class UMIVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_size, in_dim=4545, img_dims=None)

    def _load_synth(self, dataset: UmiDataset, train: bool = True) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True, shuffle=train)

    def create_loaders(self, x, y) -> Tuple[DataLoader, DataLoader]:
        dataset = UmiDataset(x, y)

        train_loader = self._load_synth(dataset, train=True)

        return train_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return -Normal(x_mb_, torch.ones_like(x_mb_)).log_prob(x_mb)


# cc = UMIVaeDataset(batch_size=128)
# tt = cc.create_loaders(x, y)
#
# for i, data in enumerate(tt):
#     print(i)