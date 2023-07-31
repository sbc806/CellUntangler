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

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from ...data import VaeDataset
from ..stats import BatchStats, BatchStatsFloat
from ..components import Component


#TODO: add mask
class Reparametrized:

    def __init__(self, q_z: Distribution, p_z: Distribution, z: Tensor, data: Tuple[Tensor, ...]) -> None:
        self.q_z = q_z
        self.p_z = p_z
        self.z = z
        self.data = data


Outputs = Tuple[List[Reparametrized], Tensor, Tensor]


class ModelVAE(torch.nn.Module):

    def __init__(self, h_dim: int, components: List[Component],
                 mask,
                 dataset: VaeDataset,
                 scalar_parametrization: bool,
                 use_relu: bool) -> None:
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param components: dimension of the latent representation (spherical, hyperbolic, euclidean)
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.components = nn.ModuleList(components)

        self.use_relu = use_relu
        if self.use_relu:
          print("Using relu in forward() and log_likelihood.")
        else:
          print("Not using relu in forward() and log_likelihood().")

        self.mask = mask
        self.num_gene = torch.sum(self.mask > 0, 1)

        dim_all = [i.dim for i in self.components]
        dim_z = sum(dim_all)

        mask_z = np.zeros(dim_z)
        mask_z[:dim_all[0]] = 1
        self.mask_z = torch.tensor(mask_z)

        self.reconstruction_loss = dataset.reconstruction_loss  # dataset-dependent, not implemented

        self.total_z_dim = sum(component.dim for component in components)
        for component in components:
            component.init_layers(h_dim, scalar_parametrization=scalar_parametrization)

    def to(self, device: torch.device) -> "ModelVAE":
        self.device = device
        return super().to(device)

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, concat_z: Tensor) -> Tensor:
        raise NotImplementedError

    # it's the forward function that defines the network structure
    def forward(self, x: Tensor) -> Outputs:
        reparametrized = []

        for i, component in enumerate(self.components):
            x_mask = x * self.mask[i]

            # Normalization is important for PCA, does not so for NN?
            if i < 1:
                x_mask = torch.nn.functional.normalize(x_mask, p=2, dim=-1)
            x_encoded = self.encode(x_mask)

            q_z, p_z, _ = component(x_encoded)
            z, data = q_z.rsample_with_parts()

            if self.use_relu:
                if 0 == i:
                    z = torch.cat((torch.relu(z[..., 0:1]), z[..., 1:]), dim=1)

            reparametrized.append(Reparametrized(q_z, p_z, z, data))

        concat_z = torch.cat(tuple(x.z for x in reparametrized), dim=-1)
        mu1, sigma_square1 = self.decode(concat_z * self.mask_z)
        mu1 = mu1[:, :self.num_gene[0]]
        sigma_square1 = sigma_square1[:self.num_gene[0]]

        mu, sigma_square = self.decode(concat_z)
        mu = torch.cat((mu1, mu[:, self.num_gene[0]:]), dim=-1)
        sigma_square = torch.cat(
            (sigma_square1, sigma_square[self.num_gene[0]:]), dim=-1)

        return reparametrized, concat_z, mu, sigma_square

    def log_likelihood(self, x: Tensor, n: int = 500) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Mini-batch of inputs.
        :param n: Number of MC samples
        :return: Monte Carlo estimate of log-likelihood.
        """
        sample_shape = torch.Size([n])
        batch_size = x.shape[0]
        prob_shape = torch.Size([n, batch_size])

        library_size = torch.sum(x, dim=1)

        log_p_z = torch.zeros(prob_shape, device=x.device)
        log_q_z_x = torch.zeros(prob_shape, device=x.device)
        zs = []

        x1 = torch.log1p(x)
        for i, component in enumerate(self.components):
            x_mask = x1 * self.mask[i]
            if i < 1:
                x_mask = torch.nn.functional.normalize(x_mask, p=2, dim=0)
            x_encoded = self.encode(x_mask)

            q_z, p_z, z_params = component(x_encoded)

            # Numerically more stable.
            z, log_q_z_x_, log_p_z_ = component.sampling_procedure.rsample_log_probs(sample_shape, q_z, p_z)

            if self.use_relu:
                if 0 == i:
                    z = torch.cat((torch.relu(z[..., 0:1]), z[..., 1:]), dim=1)

            zs.append(z)

            log_p_z += log_p_z_
            log_q_z_x += log_q_z_x_

        concat_z = torch.cat(zs, dim=-1)
        mu_, sigma_squrare_ = self.decode(concat_z)
        mu_ = mu_ * library_size[:, None]

        x_orig = x.repeat((n, 1, 1))

        # log_p_x_z = -self.reconstruction_loss(mu_, x_orig).sum(dim=-1)
        log_p_x_z = self.log_likelihood_nb(x_orig, mu_, sigma_squrare_)

        assert log_p_x_z.shape == log_p_z.shape
        assert log_q_z_x.shape == log_p_z.shape

        joint = (log_p_x_z + log_p_z - log_q_z_x)
        log_p_x = joint.logsumexp(dim=0) - np.log(n)

        assert log_q_z_x.shape == log_p_z.shape
        mi = (log_q_z_x - log_p_z).logsumexp(dim=0) - np.log(n)

        mean_z = torch.mean(concat_z, dim=1, keepdim=True)
        mean_x = torch.mean(x_orig, dim=1, keepdim=True)
        cov_norm = torch.bmm((x - mean_x).transpose(1, 2), concat_z - mean_z).mean(dim=0).norm()

        return log_p_x, mi, cov_norm

    def compute_batch_stats(self,
                            x_mb: Tensor,
                            x_mb_: Tensor,
                            sigma_square_: Tensor,
                            reparametrized: List[Reparametrized],
                            beta: float,
                            likelihood_n: int = 0) -> BatchStats:

        # Coupled
        # For each component
        #
        # bce = self.reconstruction_loss(x_mb_, x_mb).sum(dim=-1)
        bce = self.log_likelihood_nb(x_mb, x_mb_, sigma_square_)

        assert torch.isfinite(bce).all()
        # assert (bce >= 0).all()

        component_kl = []
        weight = [1.0, 1.0]
        for i, (component, r) in enumerate(zip(self.components, reparametrized)):
            kl_comp = component.kl_loss(r.q_z, r.p_z, r.z, r.data) * weight[i]

            assert torch.isfinite(kl_comp).all()
            component_kl.append(kl_comp)

        log_likelihood = None
        mi = None
        cov_norm = None
        if likelihood_n:
            log_likelihood, mi, cov_norm = self.log_likelihood(x_mb, n=likelihood_n)

        return BatchStats(bce, component_kl, beta, log_likelihood, mi, cov_norm)

    def train_step(self, optimizer: torch.optim.Optimizer, x_mb: Tensor,
                   beta: float) -> Tuple[BatchStatsFloat, Outputs]:
        optimizer.zero_grad()

        library_size = torch.sum(x_mb, dim=1)

        x_mb = x_mb.to(self.device)
        reparametrized, concat_z, x_mb_, sigma_square_ = self(torch.log1p(x_mb))

        x_mb_ = x_mb_ * library_size[:, None]

        assert x_mb_.shape == x_mb.shape
        batch_stats = self.compute_batch_stats(x_mb, x_mb_, sigma_square_,
                                               reparametrized, likelihood_n=0, beta=beta)

        loss = -batch_stats.elbo  # Maximize elbo instead of minimizing it.
        assert torch.isfinite(loss).all()
        loss.backward()

        c_params = [v for k, v in self.named_parameters() if "curvature" in k]
        if c_params:  # TODO: Look into this, possibly disable it.
            torch.nn.utils.clip_grad_norm_(c_params, max_norm=1.0, norm_type=2)  # Enable grad clip?
        optimizer.step()

        return batch_stats.convert_to_float(), (reparametrized, concat_z, x_mb_)

    def log_likelihood_nb(self, x, mu, sigma, eps=1e-16):

        log_mu_sigma = torch.log(mu + sigma + eps)

        ll = torch.lgamma(x + sigma) - torch.lgamma(sigma) - \
             torch.lgamma(x + 1) + sigma * torch.log(sigma + eps) - \
             sigma * log_mu_sigma + x * torch.log(mu + eps) - x * log_mu_sigma

        return torch.sum(ll, dim=-1)

