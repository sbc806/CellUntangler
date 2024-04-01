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

from ml_collections import ConfigDict

EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


class NBVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component],
                 mask,
                 dataset: VaeDataset,
                 config: ConfigDict) -> None:
        super().__init__(h_dim,
                         components,
                         mask,
                         dataset,
                         config)

        self.in_dim = dataset.in_dim
        self.h_dim = h_dim

        n_batch = config.n_batch
        if type(n_batch) != list:
            n_batch = [n_batch]
        self.n_batch = n_batch
        self.batch_invariant = config.batch_invariant

        total_num_of_batches = sum(self.n_batch)
        self.total_num_of_batches = total_num_of_batches
        print(f"{self.n_batch} in nb_vae.py")
        print(f"batch_invariant: {self.batch_invariant}")
        print(f"total_num_of_batches: {self.total_num_of_batches}")
        # multi-layer
        # http://adamlineberry.ai/vae-series/vae-code-experiments
        
        self.activation = config.activation
        print(f"self.activation: {self.activation}")

        input_dim = dataset.in_dim
        if not self.batch_invariant:
            input_dim = input_dim + self.total_num_of_batches
        print('self.in_dim:', self.in_dim)
        print("input_dim:", input_dim)
        encoder_szs = [input_dim] + [128, 64, h_dim]
        encoder_layers = []
      
        for in_sz, out_sz in zip(encoder_szs[:-1], encoder_szs[1:]):
            encoder_layers.append(nn.Linear(in_sz, out_sz))
            if config.use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(out_sz, momentum=config.momentum, eps=config.eps))
            # encoder_layers.append(nn.BatchNorm1d(out_sz, momentum=0.99, eps=0.001))
            if self.activation == "relu":
                encoder_layers.append(nn.ReLU())
            elif self.activation == "leaky_relu":
                encoder_layers.append(nn.LeakyReLU())
            elif self.activation == "tanh":
                encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(nn.GELU())
            # nn.BatchNorm1d(out_sz, momentum=0.01, eps=0.001)
            # encoder_layers.append(nn.BatchNorm1d(out_sz, momentum=0.99, eps=0.001))

        self.encoder = nn.Sequential(*encoder_layers)

        # construct the decoder
        hidden_sizes = [self.total_z_dim + self.total_num_of_batches] + [64, 128]
        decoder_layers = []
        for in_sz, out_sz in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            decoder_layers.append(nn.Linear(in_sz, out_sz))
            if config.use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(out_sz, momentum=config.momentum, eps=config.eps))
            # decoder_layers.append(nn.BatchNorm1d(out_sz, momentum=0.99, eps=0.001))
            if self.activation == "relu":
                decoder_layers.append(nn.ReLU())
            elif self.activation == "leaky_relu":
                decoder_layers.append(nn.LeakyReLU())
            elif self.activation == "tanh":
                decoder_layers.append(nn.Tanh())
            else:
                decoder_layers.append(nn.GELU())
            # nn.BatchNorm1d(out_sz, momentum=0.01, eps=0.001)
            # decoder_layers.append(nn.BatchNorm1d(out_sz, momentum=0.99, eps=0.001))

        self.decoder = nn.Sequential(*decoder_layers)
        
        output_dim = dataset.in_dim
        # if not self.batch_invariant:
            # output_dim = output_dim + total_num_of_batches
        print('output_dim:', output_dim)
        self.fc_mu = nn.Linear(128, output_dim)
        self.fc_sigma = nn.Linear(128, output_dim)

        if config.init == "normal":
            self.apply(self._init_weights_normal)
        elif config.init == "xavier_uniform":
            self.apply(self._init_weights_xavier_uniform)
        elif config.init == "xavier_normal":
            self.apply(self._init_weights_xavier_normal)
        elif config.init == "he_uniform":
            self.apply(self._init_weights_he_uniform)
        elif config.init == "he_normal":
            self.apply(self._init_weights_he_normal)
        elif config.init == "custom":
            self.components[0].apply(self._init_weights_xavier_uniform)
        elif config.init == "large_z1":
            print("Using large z1 initialization")
            nn.init.normal_(self.components[0].fc_mean.weight, mean=10.0)
        elif config.init == "custom_xavier_normal":
            self.components[0].apply(self._init_weights_xavier_normal)

    def encode(self, x: Tensor, batch: Tensor) -> Tensor:
        x = x.squeeze()

        assert len(x.shape) == 2
        bs, dim = x.shape

        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)

        if not self.batch_invariant and self.total_num_of_batches != 0:
            x = torch.concat((x, batch), dim=1)
        x = self.encoder(x)

        return x.view(bs, -1)  # such that x is batch * dim, similar to reshape (no need)

    def decode(self, concat_z: Tensor, batch: Tensor):
        assert len(concat_z.shape) >= 2
        bs = concat_z.size(-2)

        if self.total_num_of_batches != 0:
            concat_z = torch.concat((concat_z, batch), dim=1)
        x = self.decoder(concat_z)

        mu = torch.nn.functional.softmax(self.fc_mu(x), -1)

        sigma_square = torch.nn.functional.softplus(self.fc_sigma(x))
        sigma_square = torch.mean(sigma_square, 0)
        sigma_square = torch.clamp(sigma_square, EPS, MAX_SIGMA_SQUARE)

        # if self.batch_invariant:
        mu = mu.view(-1, bs, self.in_dim)  # flatten
        # else:
            # mu = mu.view(-1, bs, self.in_dim+self.total_num_of_batches)

        return mu.squeeze(dim=0), sigma_square

