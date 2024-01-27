# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import itertools
import torch
from torch.autograd import Variable

from typing import Any, Dict, List, Mapping, Optional

from mrest.utils.fc_network import FCNetwork, DiscreteActionFCNetwork
from mrest.utils.actions.mixture_density_network import MixtureDensityNetwork


class MLP:
    def __init__(self, 
                 env_spec,
                 input_dim: int,
                 min_log_std=-3,
                 init_log_std=0,
                 policy_mlp_kwargs: Optional[Dict[str, Any]] = None,
                 concat_proprio: bool = True,
                 concat_language: bool = False,
                 binary_gripper_config: Optional[Dict[str, Any]] = None,
                 use_GMM_action: bool = False,
                 policy_GMM_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.input_dim = input_dim
        self.min_log_std = min_log_std

        self.concat_proprio = concat_proprio
        self.concat_language = concat_language
        self.binary_gripper_config = binary_gripper_config

        # Policy network
        # ------------------------
        self.use_GMM_action = use_GMM_action
        if use_GMM_action:
            self.model = MixtureDensityNetwork(input_dim, self.m, **policy_GMM_kwargs,)
            self.log_std = None
        else:
            self.model = FCNetwork(input_dim, self.m, **policy_mlp_kwargs)
            # make weights small
            for param in self.model.action_fc.parameters():   # last action layer only
                param.data = 1e-2 * param.data
            # self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
            self.log_std = torch.nn.Parameter(torch.ones(self.m) * init_log_std, requires_grad=True)

            # Easy access variables
            # -------------------------
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())

        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params()]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params()]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.input_dim), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.cpu().contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()
    
    def trainable_params(self):
        if self.use_GMM_action:
            return itertools.chain(self.model.parameters(), [])
        else:
            return itertools.chain(self.model.parameters(), [self.log_std])
    
    def send_to_device(self, device):
        self.model = self.model.to(device)
        if self.log_std is not None:
            self.log_std = self.log_std.to(device)
    
    def get_state_dict(self):
        if self.log_std is not None:
            return {
                'model': self.model.state_dict(),
                'log_std': self.log_std,
            }
        else:
            return { 'model': self.model.state_dict(), }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        if 'log_std' in state_dict and self.log_std is not None:
            self.log_std = state_dict['log_std']

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            raise NotImplementedError

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        return self.get_noisy_action(mean)
    
    def get_noisy_action(self, mean_action):
        if self.use_GMM_action:
            # mean_action contains (log_pi, mu, std)
            mean_action_dict = mean_action
            sampled_action = self.model.sample_from_mixture(
                mean_action_dict['log_pi'], mean_action_dict['mean_action'], mean_action_dict['sigma_action'])
            sampled_action = sampled_action.cpu().numpy().ravel()
            return [
                sampled_action,
                {
                    'evaluation': sampled_action,
                }
            ]
        else:
            noise = np.exp(self.log_std_val) * np.random.randn(self.m)
            action = mean_action + noise
            # Slightly HACKY, we should write an action wrapper that handles different action spaces.
            if self.binary_gripper_config is not None and self.binary_gripper_config.use:
                assert self.m == 4, "Assuming last index is gripper action"
                action_dist = torch.distributions.bernoulli.Bernoulli(logits=out[:, -1])
                action[action_dist.mode < 0.5, -1] = self.binary_gripper_config.low
                action[action_dist.mode > 0.5, -1] = self.binary_gripper_config.high

            return [
                action, 
                {
                    'mean': mean_action,
                    'log_std': self.log_std_val,
                    'evaluation': mean_action,
                },
            ]
    
    def forward(self, observations: Mapping[str, Any], **kwargs):
        img_z = observations['img_z']
        proprio_z = observations['proprio_z']
        if self.concat_proprio:
            policy_inp = torch.cat([img_z, proprio_z], dim=1)
        else:
            policy_inp = img_z
        
        if self.concat_language:
            task_how_embd = observations['task_how_embd']
            policy_inp = torch.cat([policy_inp, task_how_embd], dim=1)
        
        if self.use_GMM_action:
            log_pi, mu, sigma = self.model(policy_inp, force_cpu=False)
            return dict(log_pi=log_pi, mean_action=mu, sigma_action=sigma)

        else:
            action = self.model(policy_inp, force_cpu=False)
            return action

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)



class DiscreteMLP:
    def __init__(self, 
                 env_spec,
                 input_dim: int,
                 action_dims: List[int],
                 policy_mlp_kwargs: Optional[Dict[str, Any]] = None,
                 concat_proprio: bool = True,
                 concat_language: bool = False,
                 binary_gripper_config: Optional[Dict[str, Any]] = None,):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        from stable_baselines3.common.distributions import MultiCategoricalDistribution
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.input_dim = input_dim
        self.action_dims = action_dims

        self.action_dist = MultiCategoricalDistribution(action_dims)

        self.concat_proprio = concat_proprio
        self.concat_language = concat_language
        self.binary_gripper_config = binary_gripper_config

        # Policy network
        # ------------------------
        self.model = DiscreteActionFCNetwork(input_dim, self.action_dims, **policy_mlp_kwargs)
        # make weights small
        for param in self.model.action_fcs.parameters():   # last action layer only
           param.data = 1e-2 * param.data

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.cpu().contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()
    
    def trainable_params(self):
        return itertools.chain(self.model.parameters())
    
    def send_to_device(self, device):
        self.model = self.model.to(device)
        self.log_std = self.log_std.to(device)
        self.old_model = self.old_model.to(device)
        self.old_log_std = self.old_log_std.to(device)
    
    def get_state_dict(self):
        return {
            'model': self.model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        action_info = self.model(self.obs_var).data.numpy().ravel()
        return self.get_sampled_action(action_info)
    
    def get_sampled_action(self, action_info: Mapping[str, Any]):
        # We don't do any sampling since we use BC only
        action_dist = action_info['dist']
        mean_action = action_dist.mode()

        return [
            mean_action, 
            {
                'mean': mean_action,
                'action_dist': action_dist,
                'evaluation': mean_action,
            },
        ]
    
    def forward(self, observations: Mapping[str, Any], **kwargs):
        img_z = observations['img_z']
        proprio_z = observations['proprio_z']
        if self.concat_proprio:
            policy_inp = torch.cat([img_z, proprio_z], dim=1)
        else:
            policy_inp = img_z
        
        if self.concat_language:
            task_how_embd = observations['task_how_embd']
            policy_inp = torch.cat([policy_inp, task_how_embd], dim=1)
        
        action_logits = self.model(policy_inp, force_cpu=False)
        dist = self.action_dist.proba_distribution(action_logits)

        return {
            'logits': action_logits,
            'dist': dist,
        }
