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

# Tensorflow implementation of HSIC
# Refers to original implementations
# https://github.com/kacperChwialkowski/HSIC
# https://cran.r-project.org/web/packages/dHSIC/index.html


from ml_collections import ConfigDict
from scipy.special import gamma
import numpy as np
import math as math


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
                 config: ConfigDict) -> None:
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param components: dimension of the latent representation (spherical, hyperbolic, euclidean)
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.components = nn.ModuleList(components)

        self.config = config

        self.use_btcvae = config.use_btcvae
        self.btcvae_beta = config.btcvae_beta
        self.dataset_size = config.dataset_size
        print("self.use_btcvae:",self.use_btcvae)
        print("self.btcvae_beta:",self.btcvae_beta)
        print("self.dataset_size:",self.dataset_size)

        self.reconstruction_term_weight = config.reconstruction_term_weight

        n_batch = config.n_batch
        if type(n_batch) != list:
            n_batch = [n_batch]
        self.n_batch = n_batch
        print(f"{self.n_batch} in vae.py")
        self.total_num_of_batches = sum(self.n_batch)
        self.zero_batch = config.zero_batch

        self.use_relu = config.use_relu
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
        
        mask_z1=np.zeros(dim_z)
        mask_z1[dim_all[0]:]=1
        self.mask_z1=torch.tensor(mask_z1)

        self.activation = config.activation
        print(f"self.activation: {self.activation}")
        self.use_hsic = config.use_hsic
        print(f"use_hsic: {self.use_hsic}")
        self.hsic_weight = config.hsic_weight
        print(f"hsic_weight: {self.hsic_weight}")
        self.reconstruction_loss = dataset.reconstruction_loss  # dataset-dependent, not implemented

        self.total_z_dim = sum(component.dim for component in components)
        for component in components:
            component.init_layers(h_dim, scalar_parametrization=config.scalar_parametrization)

    def to(self, device: torch.device) -> "ModelVAE":
        self.device = device
        return super().to(device)

    def _init_weights_normal(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing Normal weights in {}'.format(module.__class__.__name__))
            nn.init.normal_(module.weight)
            nn.init.normal_(module.bias)

    def _init_weights_xavier_uniform(self, module):
        print(module)
        if isinstance(module, torch.nn.Linear):
            print('initializing Xavier uniform weights in {}'.format(module.__class__.__name__))
            nn.init.xavier_uniform_(module.weight, gain=self.config.gain)
            module.bias.data.fill_(0.01)

    def _init_weights_xavier_normal(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing Xavier Normal weights in {}'.format(module.__class__.__name__))
            nn.init.xavier_normal_(module.weight, gain=self.config.gain)
            module.bias.data.fill_(0.01)

    def _init_weights_he_uniform(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing He uniform weights in {}'.format(module.__class__.__name__))
            nn.init.kaiming_uniform_(module.weight, nonlinearity=self.activation)
            module.bias.data.fill_(0.01)

    def _init_weights_he_normal(self, module):
        if isinstance(module, torch.nn.Linear):
            print('initializing He Normal weights in {}'.format(module.__class__.__name__))
            nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation)
            module.bias.data.fill_(0.01)

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, concat_z: Tensor) -> Tensor:
        raise NotImplementedError

    # it's the forward function that defines the network structure
    def forward(self, x: Tensor, batch: Tensor) -> Outputs:
        reparametrized = []
        all_z_params = []
        
        # Create batch or batches encoding
        if self.total_num_of_batches != 0:
            if len(self.n_batch) > 1:
                self.batch = self.multi_one_hot(batch, self.n_batch)
            else:
                if self.zero_batch and self.n_batch[0] == 1:
                    self.batch = nn.functional.one_hot(batch[:, 0], self.n_batch[0])-1
                else:
                    self.batch = nn.functional.one_hot(batch[:, 0], self.n_batch[0])
        else:
            self.batch = None
        
        # Output batch for viewing
        # if self.config.print_batch:
            # print(batch)
            # print(self.batch.shape)
            # print(self.batch)
        
        for i, component in enumerate(self.components):
            x_mask = x * self.mask[i]

            # Normalization is important for PCA, does not so for NN?
            # if i < 1:
                # x_mask = torch.nn.functional.normalize(x_mask, p=2, dim=-1)
            # if i == 0 and self.n_batch[0] > 1:
                # self.batch = torch.zeros(self.batch.shape)
            # else:
                # self.batch = nn.functional.one_hot(batch[:, 0], self.n_batch[0])
            # print(i, self.batch)
            if i == 0 and len(self.n_batch) > 1:
                self.batch = torch.zeros(self.batch.shape)
            elif i > 0 and len(self.n_batch) > 1:
                self.batch = self.multi_one_hot(batch, self.n_batch)
            
            if self.config.print_batch:
                print(batch)
                print(self.batch.shape)
                print(self.batch)
            
            x_encoded = self.encode(x_mask, self.batch)

            q_z, p_z, z_params = component(x_encoded)
            z, data = q_z.rsample_with_parts()
            # print(f"z_mean_h.shape: {z_params[0].shape}")
            # print(f"std.shape: {z_params[1].shape}")
            if self.use_relu:
                print("Using relu")
                if 0 == i:
                    z = torch.cat((torch.relu(z[..., 0:1]), z[..., 1:]), dim=1)

            reparametrized.append(Reparametrized(q_z, p_z, z, data))
            all_z_params.append(z_params)

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
        
        concat_z_params = torch.cat(tuple(z_params[0] for z_params in all_z_params), dim=-1)

        return reparametrized, concat_z, mu, sigma_square, concat_z_params, mu1, sigma_square1

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
                if self.n_batch[0] == 1 and self.zero_batch:
                    self.batch = self.batch - 1
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
        full_bce = self.log_likelihood_nb(x_mb, x_mb_, sigma_square_)
        
        first_bce=torch.sum(full_bce[:,0:self.num_gene[0]],dim=-1)
        second_bce=torch.sum(full_bce[:,self.num_gene[0]:],dim=-1)

        bce = torch.sum(full_bce, dim=-1)
        assert torch.isfinite(bce).all()
        # assert (bce >= 0).all()

        if self.use_btcvae:
            z1_samples = reparametrized[0].z
            z1_data = reparametrized[0].data
            z1_prior_dist = reparametrized[0].p_z
            z1_q_dist = reparametrized[0].p_z

            z2_samples = reparametrized[1].z
            z2_data = reparametrized[1].data
            z2_prior_dist = reparametrized[1].p_z
            z2_q_dist = reparametrized[1].q_z
            # Adjusted ELBO
            log_px = bce
        
            dataset_size = self.dataset_size
            batch_size = z1_samples.shape[0]

            logpz2 = z2_prior_dist.log_prob_individual(z2_samples).view(batch_size, -1).sum(1)
            logqz2_condx = z2_q_dist.log_prob_individual(z2_samples).view(batch_size, -1).sum(1)
            
            expanded_z2_samples = z2_samples.unsqueeze(1).repeat((1, batch_size, 1))
            _logqz2 = z2_q_dist.log_prob_individual(expanded_z2_samples)
            print("_logqz2.shape:",_logqz2.shape)

            # Minibatch weighted sampling
            logqz2_prodmarginals = (torch.logsumexp(_logqz2, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz2 = torch.logsumexp(_logqz2.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size)

            lamb = 0
            z2_kl = (logqz2_condx - logqz2) + \
                self.btcvae_beta * (logqz2 - logqz2_prodmarginals) + \
                (1 - lamb) * (logqz2_prodmarginals - logqz2)

            component_kl = []
            r1 = reparametrized[0]
            component_1_kl = self.components[0].kl_loss(r1.q_z, r1.p_z, r1.z, r1.data)
            assert torch.isfinite(component_1_kl).all()
            assert torch.isfinite(z2_kl).all()
            component_kl.append(component_1_kl)
            component_kl.append(z2_kl)
        else:
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

        batch_hsic = None
        # if self.use_hsic:
            # hsic = self.calculate_hsic(reparametrized[0].z, reparametrized[1].z) * 1000
            # z1_poincare = lorentz_to_poincare(reparametrized[0].z, self.components[0].manifold.curvature)
            # batch_hsic = hsic(z1_poincare, reparametrized[1].z) * self.hsic_weight
        # print(f"batch_hsic: {batch_hsic}")
        # return BatchStats(bce, component_kl, beta, log_likelihood, mi, cov_norm, batch_hsic, self.reconstruction_term_weight)
        return BatchStats(bce, component_kl, beta, log_likelihood, first_bce, second_bce, batch_hsic, self.reconstruction_term_weight)

    def train_step(self, optimizer: torch.optim.Optimizer, x_mb: Tensor, y_mb: Tensor,
                   beta: float) -> Tuple[BatchStatsFloat, Outputs]:
        optimizer.zero_grad()

        library_size = torch.sum(x_mb, dim=1)

        x_mb = x_mb.to(self.device)
        y_mb = y_mb.to(self.device)
        reparametrized, concat_z, x_mb_, sigma_square_, concat_z_params, _, _ = self(torch.log1p(x_mb), y_mb)

        x_mb_ = x_mb_ * library_size[:, None]

        assert x_mb_.shape == x_mb.shape
        batch_stats = self.compute_batch_stats(x_mb, x_mb_, y_mb, sigma_square_,
                                               reparametrized, likelihood_n=0, beta=beta)

        if self.config.use_hsic:
            batch_hsic=hsic_hyperbolic(lorentz_to_poincare(concat_z[:,0:3],-2),lorentz_to_poincare(concat_z[:,3:],-1),-2,-1)
            loss=-(batch_stats.elbo-batch_hsic*self.config.hsic_weight)
            print(batch_hsic)
        elif self.config.use_average_hsic:
            batch_hsic=hsic_hyperbolic(lorentz_to_poincare(concat_z[:,0:3],-2),lorentz_to_poincare(concat_z[:,3:],-1),-2,-1)
            loss=-(batch_stats.elbo-batch_hsic/self.config.dataset_size*self.config.hsic_weight)
            print(batch_hsic/self.config.dataset_size*self.config.hsic_weight)
        else:
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

        # return torch.sum(ll, dim=-1)
        return ll

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

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel, 
    this can be used to select the bandwidth according to the median heuristic
    """
    print(d)
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)
    
def K(x1, x2, gamma=1.): 
    dist_table = torch.unsqueeze(x1, 0) - torch.unsqueeze(x2, 1)
    return torch.exp(-gamma * torch.sum(dist_table **2, dim=2)).T

def hsic(z, s):
    
    # use a gaussian RBF for every variable
      
    d_z = list(z.shape)[1]
    d_s = list(s.shape)[1]
    
    zz = K(z, z, gamma= bandwidth(d_z))
    ss = K(s, s, gamma= bandwidth(d_s))
        
        
    hsic = 0
    hsic += torch.mean(zz * ss) 
    hsic += torch.mean(zz) * torch.mean(ss)
    hsic -= 2 * torch.mean( torch.mean(zz, dim=1) * torch.mean(ss, dim=1) )
    return torch.sqrt(hsic)

def K_hyperbolic(x1, x2, gamma=1., curvature=-1):
    x1 = fd_distance(x1, curvature=curvature)
    x2 = fd_distance(x2, curvature=curvature)
    return K(x1, x2, gamma=gamma)

def hsic_hyperbolic(z, s, curvature_1=-1, curvature_2=-1):
    d_z = list(z.shape)[1]
    d_s = list(s.shape)[1]

    zz = K_hyperbolic(z, z, gamma=bandwidth(d_z), curvature=curvature_1)
    ss = K_hyperbolic(s, s, gamma=bandwidth(d_s), curvature=curvature_2)

    hsic = 0
    hsic += torch.mean(zz * ss)
    hsic += torch.mean(zz) ** torch.mean(ss)
    hsic -= 2 * torch.mean(torch.mean(zz, dim=1) * torch.mean(ss, dim=1))
    return torch.sqrt(hsic)

def hsic_mixed(z, s , curvature=-1):
    d_z = list(z.shape)[1]
    d_s = list(s.shape)[1]

    zz = K_hyperbolic(z, z, gamma=bandwidth(d_z), curvature=curvature)
    ss = K(s, s, gamma=bandwidth(d_s))

    hsic = 0
    hsic += torch.mean(zz * ss)
    hsic += torch.mean(zz) * torch.mean(ss)
    hsic -= 2 * torch.mean(torch.mean(zz, dim=1) * torch.mean(ss, dim=1))
    return torch.sqrt(hsic)

def fd_distance(z, curvature=-1):
    z_norm = torch.norm(z, dim=1)
    abs_curvature = abs(curvature)
    coefficient = torch.atanh(math.sqrt(abs_curvature)*z_norm)/(math.sqrt(abs_curvature)*z_norm)
    distance = coefficient.unsqueeze(-1).expand(z.shape)*z
    return distance

def lorentz_to_poincare(embeddings, curvature):
    return embeddings[:, 1:] / (1 + math.sqrt(abs(curvature)) * embeddings[:, 0:1])