import numpy as np

import torch
import torch.nn as nn
from typing import Any, Optional, List, Mapping

from enum import Enum, auto
from omegaconf import ListConfig
from mrest.utils.torch_utils import get_nonlinearity_from_str


# Code taken from https://github.com/tonyduan/mixture-density-network
# with some minor modifications.


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 num_components: int,
                 hidden_sizes: Optional[List[int]] = None,
                 nonlinearity = 'tanh',   # either 'tanh' or 'relu'
                 noise_type=NoiseType.DIAGONAL,
                 fixed_noise_level=None,
                 use_batchnorm: bool = False,
                 use_tanh_action: bool = False,
                 predict_future: Optional[Mapping[str, Any]] = None,):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, num_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level

        self.predict_future = predict_future
        if predict_future is not None and predict_future.use:
            total_actions = predict_future.total_actions
            # TODO: Since we only predict one gripper action still
            self.dim_out = (self.dim_out - 1) * total_actions + 1
            dim_out = (dim_out - 1) * total_actions + 1

        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * num_components,
            NoiseType.ISOTROPIC: num_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]

        if isinstance(hidden_sizes, list) or isinstance(hidden_sizes, ListConfig):
            hidden_sizes = tuple(hidden_sizes)

        self.pi_layer_sizes = (dim_in,) + hidden_sizes + (num_components,)
        self.normal_layer_sizes = (dim_in,) + hidden_sizes + (dim_out * num_components + num_sigma_channels,)

        pi_modules = []
        normal_modules = []
        for i in range(len(self.pi_layer_sizes) - 1):
            pi_modules.append(nn.Linear(self.pi_layer_sizes[i], self.pi_layer_sizes[i + 1]))
            if i < len(self.pi_layer_sizes) - 2:
                pi_modules.append(get_nonlinearity_from_str(nonlinearity))

            normal_modules.append(nn.Linear(self.normal_layer_sizes[i], self.normal_layer_sizes[i + 1]))
            if i < len(self.pi_layer_sizes) - 2:
                normal_modules.append(get_nonlinearity_from_str(nonlinearity))
        
        self.pi_network = nn.Sequential(*pi_modules)
        self.normal_network = nn.Sequential(*normal_modules)

        # Batch Norm Layers
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = torch.nn.BatchNorm1d(dim_in)
    
    def forward(self, x, eps=1e-6, **kwargs):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #

        ## BATCHNORM
        if self.use_batchnorm:
            x = self.bn(x)

        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)

        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)

        return log_pi, mu, sigma
    
    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        return self.sample_from_mixture(log_pi, mu, sigma)
    
    def sample_from_mixture(self, log_pi, mu, sigma):
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(log_pi), 1).to(log_pi)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        # randn_normal = mu + sigma * N(0, 1)
        use_mean = False
        if use_mean:
            rand_normal = torch.ones_like(mu) * 0.1 + mu
        else:
            rand_normal = torch.randn_like(mu) * sigma + mu

        # samples = torch.gather(rand_normal, index=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        samples = torch.gather(
            rand_normal,
            index=rand_pi.unsqueeze(-1).repeat(1, 1,rand_normal.size(2)),
            dim=1).squeeze(dim=1)
        return samples


if __name__ == '__main__':
    dim_in = 8
    dim_out = 4
    num_components = 3
    mdn = MixtureDensityNetwork(
        dim_in=dim_in,
        dim_out=dim_out,
        num_components=num_components,
        hidden_sizes=[256],
        nonlinearity = 'tanh',   # either 'tanh' or 'relu'
        noise_type=NoiseType.DIAGONAL,
    )

    bs = 2
    x = torch.rand(bs, dim_in)
    log_pi, mu, sigma = mdn.forward(x)
    samples = mdn.sample(x)
    breakpoint()
