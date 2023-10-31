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
                 use_relu: bool,
                 n_batch: int = 0,
                 use_hsic: bool = False) -> None:
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param components: dimension of the latent representation (spherical, hyperbolic, euclidean)
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.components = nn.ModuleList(components)

        if type(n_batch) != list:
            n_batch = [n_batch]
        self.n_batch = n_batch
        print(f"{self.n_batch} in vae.py")
        self.total_num_of_batches = sum(self.n_batch)

        self.use_relu = use_relu
        if self.use_relu:
          print("Using relu in forward() and log_likelihood.")
        else:
          print("Not using relu in forward() and log_likelihood().")
        # print("Uncommented out the normalization step in forward() and log_likelihood().")
        print("Commented out the normalization step in forward() and log_likelihood().")
        self.mask = mask
        self.num_gene = torch.sum(self.mask > 0, 1)
        print('self.num_gene:', self.num_gene)

        dim_all = [i.dim for i in self.components]
        dim_z = sum(dim_all)

        mask_z = np.zeros(dim_z)
        mask_z[:dim_all[0]] = 1
        self.mask_z = torch.tensor(mask_z)
        
        self.use_hsic = use_hsic
        print(f"use_hsic: {self.use_hsic}")
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
    def forward(self, x: Tensor, batch: Tensor) -> Outputs:
        reparametrized = []

        if self.total_num_of_batches != 0:
            if len(self.n_batch) > 1:
                self.batch = self.multi_one_hot(batch, self.n_batch)
            else:
                self.batch = nn.functional.one_hot(batch[:, 0], self.n_batch[0])
        else:
            self.batch = None
        # print(batch)
        # print(self.batch.shape)
        # print(self.batch)
        for i, component in enumerate(self.components):
            x_mask = x * self.mask[i]

            # Normalization is important for PCA, does not so for NN?
            # if i < 1:
                # x_mask = torch.nn.functional.normalize(x_mask, p=2, dim=-1)
            x_encoded = self.encode(x_mask, self.batch)

            q_z, p_z, z_params = component(x_encoded)
            z, data = q_z.rsample_with_parts()
            # print(f"z_mean_h.shape: {z_params[0].shape}")
            # print(f"std.shape: {z_params[1].shape}")
            if self.use_relu:
                if 0 == i:
                    z = torch.cat((torch.relu(z[..., 0:1]), z[..., 1:]), dim=1)

            reparametrized.append(Reparametrized(q_z, p_z, z, data))

        concat_z = torch.cat(tuple(x.z for x in reparametrized), dim=-1)
        mu1, sigma_square1 = self.decode(concat_z * self.mask_z, self.batch)
        mu1 = mu1[:, :self.num_gene[0]]
        sigma_square1 = sigma_square1[:self.num_gene[0]]

        # print("Using new_reparametrized and new_concat_z")
        # new_reparametrized = [self.compute_r2(x)] + reparametrized[1:]        
        # new_concat_z = torch.cat(tuple(x.z for x in new_reparametrized), dim=-1)

        mu, sigma_square = self.decode(concat_z, self.batch)
        # mu, sigma_square = self.decode(new_concat_z)
        mu = torch.cat((mu1, mu[:, self.num_gene[0]:]), dim=-1)
        sigma_square = torch.cat(
            (sigma_square1, sigma_square[self.num_gene[0]:]), dim=-1)
        # print(f"mu.shape: {mu.shape}")
        # print(f"sigma_square.shape: {sigma_square.shape}")
        return reparametrized, concat_z, mu, sigma_square

    @torch.no_grad()
    def compute_r2(self, x):
        x_mask = x * self.mask[0]
        x_encoded = self.encode(x_mask)

        q_z, p_z, _ = self.components[0](x_encoded)
        z, data = q_z.rsample_with_parts()
        return Reparametrized(q_z, p_z, z, data)

    def log_likelihood(self, x: Tensor, batch: Tensor, n: int = 500) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Mini-batch of inputs.
        :param n: Number of MC samples
        :return: Monte Carlo estimate of log-likelihood.
        """
        print('Computing log_likelihood!!!')
        sample_shape = torch.Size([n])
        batch_size = x.shape[0]
        prob_shape = torch.Size([n, batch_size])

        library_size = torch.sum(x, dim=1)

        log_p_z = torch.zeros(prob_shape, device=x.device)
        log_q_z_x = torch.zeros(prob_shape, device=x.device)
        zs = []

        x1 = torch.log1p(x)
        if self.total_num_of_batches != 0:
            if len(self.n_batch) > 1:
                self.batch = self.multi_one_hot(batch, self.n_batch)
            else:
                self.batch = nn.functional.one_hot(batch[:, 0], self.n_batch[0])
        else:
            self.batch = None
        # print('self.batch:',self.batch)
        for i, component in enumerate(self.components):
            x_mask = x1 * self.mask[i]
            # if i < 1:
                # x_mask = torch.nn.functional.normalize(x_mask, p=2, dim=0)
            x_encoded = self.encode(x_mask, self.batch)

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
        mu_, sigma_square_ = self.decode(concat_z, self.batch)
        mu_ = mu_ * library_size[:, None]
        # Copied from forward() below
        # mu1, sigma_square1 = self.decode(concat_z * self.mask_z, self.batch)
        # mu1 = mu1[:, :self.num_gene[0]]
        # sigma_square1 = sigma_square1[:self.num_gene[0]]

        # print("Using new_reparametrized and new_concat_z")
        # new_reparametrized = [self.compute_r2(x)] + reparametrized[1:]        
        # new_concat_z = torch.cat(tuple(x.z for x in new_reparametrized), dim=-1)

        # mu, sigma_square = self.decode(concat_z, self.batch)
        # mu, sigma_square = self.decode(new_concat_z)
        # mu = torch.cat((mu1, mu[:, self.num_gene[0]:]), dim=-1)
        # sigma_square = torch.cat(
            # (sigma_square1, sigma_square[self.num_gene[0]:]), dim=-1)
        # mu_ = mu * library_size[:, None]
        # sigma_square_ = sigma_square
        print('x.shape:',x.shape)
        x_orig = x.repeat((n, 1, 1))

        # log_p_x_z = -self.reconstruction_loss(mu_, x_orig).sum(dim=-1)
        log_p_x_z = self.log_likelihood_nb(x_orig, mu_, sigma_square_)

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
                            batch: Tensor,
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
            # print(kl_comp.shape)
            assert torch.isfinite(kl_comp).all()
            component_kl.append(kl_comp)

        log_likelihood = None
        mi = None
        cov_norm = None
        if likelihood_n:
            log_likelihood, mi, cov_norm = self.log_likelihood(x_mb, batch, n=likelihood_n)

        hsic = None
        if self.use_hsic:
            hsic = self.calculate_hsic(reparametrized[0].z, reparametrized[1].z) * 1000
        # print(f"hsic: {hsic}")
        return BatchStats(bce, component_kl, beta, log_likelihood, mi, cov_norm, hsic)

    def train_step(self, optimizer: torch.optim.Optimizer, x_mb: Tensor, y_mb: Tensor,
                   beta: float) -> Tuple[BatchStatsFloat, Outputs]:
        optimizer.zero_grad()

        library_size = torch.sum(x_mb, dim=1)

        x_mb = x_mb.to(self.device)
        y_mb = y_mb.to(self.device)
        reparametrized, concat_z, x_mb_, sigma_square_ = self(torch.log1p(x_mb), y_mb)

        x_mb_ = x_mb_ * library_size[:, None]

        assert x_mb_.shape == x_mb.shape
        batch_stats = self.compute_batch_stats(x_mb, x_mb_, y_mb, sigma_square_,
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

    def multi_one_hot(self, indices, depth_list):
        one_hot_tensor = nn.functional.one_hot(indices[:,0], depth_list[0])
        if depth_list[0] == 1:
            one_hot_tensor = one_hot_tensor - 1
        for col in range(1, len(depth_list)):
            next_one_hot = nn.functional.one_hot(indices[:,col], depth_list[col])
            if col == 1:
                next_one_hot = next_one_hot - 1
            one_hot_tensor = torch.concat((one_hot_tensor, next_one_hot), dim=1)
        
        return one_hot_tensor

    def calculate_hsic(self, z1, z2):
        n = z1.shape[0]
        u_kernels = torch.zeros((n, n), dtype=torch.float64)
        v_kernels = torch.zeros((n, n), dtype=torch.float64)

        z1_gamma = torch.sqrt(calculate_median_gamma(z1)/2)
        z2_gamma = torch.sqrt(calculate_median_gamma(z2)/2)

        for i in range(0, n):
            for j in range(0, n):
                u_kernels[i][j] = self.gaussian_kernel(z1[i], z1[j], z1_gamma)
                v_kernels[i][j] = self.gaussian_kernel(z2[i], z2[j], z2_gamma)

        first_term = torch.sum(u_kernels * v_kernels) / n**2

        second_term = 0
        for i in range(0, n):
            for j in range(0, n):
                second_term = second_term + torch.sum(u_kernels[i][j] * v_kernels)
        second_term = second_term / n**4

        third_term = 0
        for i in range(0, n):
            for j in range(0, n):
                third_term = third_term + torch.sum(u_kernels[i][j] * v_kernels[i])
        third_term = third_term * 2 / n**3
        return first_term + second_term - third_term
    
    def gaussian_kernel(self, u, u_1, gamma=1.0):
        """
        A universal kernel when gamma > 1.0.
        """
        difference = u - u_1
        squared_norm = torch.linalg.vector_norm(difference, ord=2)**2
  
        return torch.exp(-gamma * squared_norm)


def calculate_median_gamma(x):
    n = x.shape[0]
    medians = torch.zeros(int((n-1)*(n-1+1)/2), dtype=torch.float64)
    count = 0
    for i in range(0, n-1):
        for j in range(i+1, n):
            medians[count] = torch.linalg.norm(x[i] - x[j], ord=2)**2
            count = count + 1
    return torch.median(medians)
