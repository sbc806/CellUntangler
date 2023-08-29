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

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from ...data import VaeDataset
from ..components import Component

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


class NBVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component],
                 mask,
                 dataset: VaeDataset,
                 scalar_parametrization: bool,
                 cell_cycle_components=[]) -> None:
        super().__init__(h_dim, components, mask, dataset, scalar_parametrization, cell_cycle_components)

        self.in_dim = dataset.in_dim
        self.h_dim = h_dim

        # multi-layer
        # http://adamlineberry.ai/vae-series/vae-code-experiments
        encoder_szs = [dataset.in_dim] + [128, 64, h_dim]
        encoder_layers = []
        for in_sz, out_sz in zip(encoder_szs[:-1], encoder_szs[1:]):
            encoder_layers.append(nn.Linear(in_sz, out_sz))
            encoder_layers.append(nn.GELU())
            # nn.BatchNorm1d(out_sz, momentum=0.01, eps=0.001)

        self.encoder = nn.Sequential(*encoder_layers)

        # construct the decoder
        hidden_sizes = [self.total_z_dim] + [64, 128]
        decoder_layers = []
        for in_sz, out_sz in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            decoder_layers.append(nn.Linear(in_sz, out_sz))
            decoder_layers.append(nn.GELU())
            # nn.BatchNorm1d(out_sz, momentum=0.01, eps=0.001)

        self.decoder = nn.Sequential(*decoder_layers)

        self.fc_mu = nn.Linear(128, dataset.in_dim)
        self.fc_sigma = nn.Linear(128, dataset.in_dim)

    def encode(self, x: Tensor) -> Tensor:
        x = x.squeeze()

        assert len(x.shape) == 2
        bs, dim = x.shape

        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)

        x = self.encoder(x)

        return x.view(bs, -1)  # such that x is batch * dim, similar to reshape (no need)

    def decode(self, concat_z: Tensor):
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        x = self.decoder(concat_z)

        mu = torch.nn.functional.softmax(self.fc_mu(x), -1)

        sigma_square = torch.nn.functional.softplus(self.fc_sigma(x))
        sigma_square = torch.mean(sigma_square, 0)
        sigma_square = torch.clamp(sigma_square, EPS, MAX_SIGMA_SQUARE)

        mu = mu.view(-1, bs, self.in_dim)  # flatten

        return mu.squeeze(dim=0), sigma_square

