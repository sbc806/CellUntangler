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
# import anndata


# https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
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
            return torch.squeeze(torch.tensor(data)), torch.tensor(self.y[i])
        else:
            return torch.tensor(data)


# https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
class UMIVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(batch_size, in_dim=253, img_dims=None)

    def _load_synth(self, dataset: UmiDataset, train: bool = True) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          num_workers=0, pin_memory=True, shuffle=train)

    def create_loaders(self, x, y) -> Tuple[DataLoader, DataLoader]:
        dataset = UmiDataset(x, y)

        train_loader = self._load_synth(dataset, train=True)

        return train_loader


# adata = anndata.read_loom('/Users/jding/work/desc/hela_select.loom')
#
# x = adata.X.todense().astype(int)
# y = adata.obs['batch'].rank(method='dense').astype(int)
#
# dd = UmiDataset(x, y)
#
# # dataloder = DataLoader(dataset=dd, batch_size=128)
# #
# # for i, data in enumerate(dataloder):
# #     print(i)
# #
# #
# from mt.data.vae_dataset import VaeDataset
# from typing import Tuple, Any




# cc = UMIVaeDataset(batch_size=128)
# tt = cc.create_loaders(x, y)
#
# for i, data in enumerate(tt):
#     print(i)