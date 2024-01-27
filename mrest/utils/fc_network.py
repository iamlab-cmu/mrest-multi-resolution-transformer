# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, List, Mapping, Optional
from omegaconf import ListConfig
from mrest.utils.torch_utils import get_nonlinearity_from_str


class FCNetwork(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 act_dim,
                 hidden_sizes: Optional[List[int]] = None,
                 nonlinearity = 'tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None,
                 use_batchnorm: bool = False,
                 use_tanh_action: bool = False,
                 predict_future: Optional[Mapping[str, Any]] = None,):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if isinstance(hidden_sizes, list) or isinstance(hidden_sizes, ListConfig):
            hidden_sizes = tuple(hidden_sizes)
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.use_batchnorm = use_batchnorm
        self.use_tanh_action = use_tanh_action
        self.use_predict_future = predict_future

        assert predict_future is None or not predict_future['use'], "Future action prediction not enabled."

        # Batch Norm Layers
        if use_batchnorm:
            self.bn = torch.nn.BatchNorm1d(obs_dim)

        # hidden layers
        modules = []
        for i in range(len(self.layer_sizes) - 2):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(get_nonlinearity_from_str(nonlinearity))
        self.pre_action_fc = nn.Sequential(*modules)
        self.action_fc = nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(
            in_shift=in_shift,
            in_scale=in_scale,
            out_shift=out_shift,
            out_scale=out_scale
        )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) \
            if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) \
            if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) \
            if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) \
            if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x, force_cpu=True):
        # Small MLP runs on CPU
        # Required for the way the Gaussian MLP class does weight saving and loading.
        if force_cpu:
            if x.is_cuda:
                out = x.to('cpu')
            else:
                out = x
        else:
            out = x
        
        ## BATCHNORM
        if self.use_batchnorm:
            out = self.bn(out)
        
        out = self.pre_action_fc(out)
        out = self.action_fc(out)
        # out = out * self.out_scale + self.out_shift

        if self.use_tanh_action:
            out = F.tanh(out)

        return out


class DiscreteActionFCNetwork(nn.Module):
    def __init__(self, 
                 obs_dim: int, 
                 act_dims: List[int],
                 hidden_sizes: Optional[List[int]] = None,
                 nonlinearity = 'tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None,
                 use_batchnorm: bool = False):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.act_dims = act_dims
        if isinstance(hidden_sizes, list) or isinstance(hidden_sizes, ListConfig):
            hidden_sizes = tuple(hidden_sizes)
        self.layer_sizes = (obs_dim, ) + hidden_sizes
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.use_batchnorm = use_batchnorm

        # Batch Norm Layers
        if use_batchnorm:
            self.bn = torch.nn.BatchNorm1d(obs_dim)

        # hidden layers
        modules = []
        for i in range(len(self.layer_sizes) - 1):
            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(get_nonlinearity_from_str(nonlinearity))
        self.pre_action_fc = nn.Sequential(*modules)
        self.action_fc = nn.Linear(self.layer_sizes[-1], sum(act_dims))


    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(
            in_shift=in_shift,
            in_scale=in_scale,
            out_shift=out_shift,
            out_scale=out_scale
        )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) \
            if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) \
            if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) \
            if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) \
            if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x, force_cpu=True):
        # Small MLP runs on CPU
        # Required for the way the Gaussian MLP class does weight saving and loading.
        if force_cpu:
            if x.is_cuda:
                out = x.to('cpu')
            else:
                out = x
        else:
            out = x
        
        ## BATCHNORM
        if self.use_batchnorm:
            out = self.bn(out)
        
        out = self.pre_action_fc(out)
        action_out = self.action_fc(out)
        return action_out
