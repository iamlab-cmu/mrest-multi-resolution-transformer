# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mrest.utils.logger import DataLog
from tqdm import tqdm
import os
from pathlib import Path

from typing import Any, Dict, List, Mapping, Optional, Union

from mrest.utils.encoder_with_proprio import EncoderWithProprio
from mrest.utils.torch_utils import (create_optimizer_with_params, create_optimizer_with_param_groups,
                                       create_schedulder_with_params)


class ScaledMSELoss(torch.nn.Module):
    def __init__(self, weights: List[float] = [1., 1., 1., 1.]):
        super(ScaledMSELoss, self).__init__()
        self.loss_wt = torch.Tensor(weights)

    def forward(self, output, target):
        if target.device != self.loss_wt.device:
            self.loss_wt = self.loss_wt.to(target.device)
        diff = (output - target) ** 2
        assert diff.size(1) == self.loss_wt.size(0)
        mean_loss = diff.mean(axis=0) * self.loss_wt
        return mean_loss.mean()


class ScaledL1Loss(torch.nn.Module):
    def __init__(self, weights: List[float] = [1., 1., 1., 1.]):
        super(ScaledL1Loss, self).__init__()
        self.loss_wt = torch.Tensor(weights)

    def forward(self, output, target):
        if target.device != self.loss_wt.device:
            self.loss_wt = self.loss_wt.to(target.device)
        diff = torch.abs(output - target)
        assert diff.size(1) == self.loss_wt.size(0)
        mean_loss = diff * self.loss_wt
        return mean_loss.mean()


class MseBinaryGripper(torch.nn.Module):
    def __init__(self, pos_scale: float = 1.0, gripper_scale: float = 1.0):
        super(MseBinaryGripper, self).__init__()
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale

        self.pos_loss = torch.nn.MSELoss()
        self.gripper_loss = torch.nn.BCEWithLogitsLoss(weight=torch.Tensor([gripper_scale]))

    def forward(self, output, target):
        pos_output, gripper_output = output[:, :-1], output[:, -1]
        pos_target, gripper_target = target[:, :-1], target[:, -1]
        binary_gripper_target = (gripper_target >= 0).type(torch.float32)

        if self.gripper_loss.weight.device != output.device:
            self.gripper_loss.weight = self.gripper_loss.weight.to(output.device)
        
        pos_loss = self.pos_loss(pos_output, pos_target)
        gripper_loss = self.gripper_loss(gripper_output, binary_gripper_target)
    
        return pos_loss + gripper_loss


class DiscreteActionLoss(torch.nn.Module):
    def __init__(self, action_dims: List[int] = [], loss_scales: List[float] = []):
        super(MseBinaryGripper, self).__init__()
        self.action_dims = action_dims
        self.loss_scales = loss_scales

        self.losses = [torch.nn.CrossEntropyLoss(loss_scales[i]) for i in range(len(loss_scales))]


    def forward(self, output, target):
        action_dist = output['dist']
        action_logits = output['logits_per_dim']

        loss = 0.0
        for logit in action_logits:
            loss += self.losses(logit.softmax(dim=1), target)
        
        return loss


class BCWithEncoderMultiTask:
    def __init__(self, 
                 train_dl,
                 val_dl,
                 policy_with_encoder: EncoderWithProprio,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer: Optional[Mapping[str, Any]] = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 set_transforms = False,
                 finetune = False,
                 proprio: Optional[int] = 0,
                 camera_name: str = 'camera0',
                 device: str = 'cuda',
                 scheduler: Optional[Mapping[str, Any]] = None,
                 grad_clip: Optional[Mapping[str, Any]] = None,
                 loss_config: Optional[Mapping[str, Any]] = None,
                 **kwargs,
                 ):

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.policy = policy_with_encoder
        self.device = device

        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.finetune = finetune
        self.proprio = proprio
        self.camera_name = camera_name

        self.train_steps = 0
        self.train_epochs = 0
        self.val_epochs = 0

        assert not set_transforms, "Not implemented"

        # construct optimizer
        if 'param_groups' in optimizer and optimizer['param_groups']['use']:
            param_groups = self.policy.get_trainable_parameters(return_param_groups=True)
            self.optimizer = create_optimizer_with_param_groups(optimizer, param_groups)
        else:
            self.optimizer = create_optimizer_with_params(optimizer, self.policy.get_trainable_parameters())

        if scheduler is not None and scheduler['use']:
            self.scheduler, scheduler_extra_dict = create_schedulder_with_params(
                scheduler, self.optimizer)
            # If True update scheduler at epochs else after every step
            self.scheduler_t_in_epochs = scheduler_extra_dict['t_in_epochs']
            self.scheduler_is_timm = scheduler_extra_dict['timm_scheduler']

        else:
            self.scheduler = None
        
        self.grad_clip = grad_clip
        self.use_grad_clip = grad_clip['use'] if grad_clip else False
        if self.use_grad_clip:
            assert grad_clip is not None
            self.grad_clip_norm = grad_clip['norm']

        # Loss criterion if required
        assert loss_config is not None
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()
            self.loss_scale = loss_config['MSE']['loss_scale']
        elif loss_type == 'weighted_L1':
            self.loss_criterion = ScaledL1Loss(weights=loss_config['weighted_L1']['weights'])
            self.loss_scale = 1.
        elif loss_type == 'weighted_mse':
            self.loss_criterion = ScaledMSELoss(weights=loss_config['weighted_mse']['weights'])
            self.loss_scale = 1.
        elif loss_type == 'mse_binary_gripper':
            self.loss_criterion = MseBinaryGripper(**loss_config['mse_binary_gripper'])
            self.loss_scale = 1.
        elif loss_type == 'discrete_actions':
            raise NotImplementedError
        elif loss_type == 'MLE':
            # Begin MLE type losses
            self.loss_scale = loss_config['MSE']['loss_scale']
        elif loss_type == 'MLE_multi_action':
            # Begin MLE type losses
            self.loss_scales = loss_config['MLE_multi_action']['loss_scales']
            self.num_actions = loss_config['MLE_multi_action']['num_actions']
        else:
            raise ValueError(f'Invalid loss: {loss_type}')

        # make logger
        if self.save_logs:
            self.logger = DataLog()
    
    def loss(self, data):
        if self.loss_type == 'MLE':
            return self.mle_loss(data)
        elif self.loss_type == 'MLE_multi_action':
            return self.mle_multi_action_loss(data)
        elif self.loss_type in ('MSE', 'weighted_mse', 'mse_binary_gripper', 'weighted_L1'):
            return self.mse_loss(data)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data):
        expert_actions = data['expert_actions'].to(self.device)

        # Policy has the encoder for image observation
        act_pi = self.policy.run_on_batch(data)

        # act_pi contains log_pi, mean_action, sigma_action

        log_pi = act_pi['log_pi']
        mean_action = act_pi['mean_action']
        sigma_action = act_pi['sigma_action']

        z_score = (expert_actions.unsqueeze(1) - mean_action) / sigma_action
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            -torch.sum(torch.log(sigma_action), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik.mean()

    def mle_multi_action_loss(self, data):
        expert_actions = data['expert_actions'].to(self.device)

        # Policy has the encoder for image observation
        act_pi = self.policy.run_on_batch(data)

        # act_pi contains log_pi, mean_action, sigma_action

        log_pi = act_pi['log_pi']
        mean_action = act_pi['mean_action']
        sigma_action = act_pi['sigma_action']

        z_score = (expert_actions.unsqueeze(1) - mean_action) / sigma_action
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            -torch.sum(torch.log(sigma_action), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)

        all_action_loglik = -0.5 * (z_score * z_score) - torch.log(sigma_action)
        loss_dict = {}
        for i in range(0, all_action_loglik.size(2) - 1, 3):
            loss_dict[f'loglik_action_{i}'] = -torch.logsumexp(
                torch.sum(all_action_loglik[:, :, i:i+3], dim=-1), dim=-1).mean().detach().cpu().numpy()

        return -loglik.mean(), loss_dict


    def mse_loss(self, data):
        expert_actions = data['expert_actions'].to(self.device)

        # Policy has the encoder for image observation
        act_pi = self.policy.run_on_batch(data)

        return self.loss_scale * self.loss_criterion(act_pi, expert_actions)
    
    def get_current_learning_rate(self) -> Optional[float]:
        '''Get current learning rate for logging.'''
        lr = None
        if self.scheduler:
            if self.scheduler_is_timm:
                lr = self.scheduler.get_epoch_values(self.train_epochs)[0]
            else:
                lr = self.scheduler.get_last_lr()[0]

        return lr
    

    def fit(self, dataloader, train: bool = True, 
            epochs: int = 1, suppress_fit_tqdm: bool = True,
            reset_policy_after_epoch: bool = False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"

        # validate_keys = all([k in data.keys() for k in [self.camera_name, "expert_actions"]])
        # assert validate_keys is True
        ts = timer.time()
        # num_samples = data[self.camera_name].shape[0]

        log_prefix = 'train_' if train else 'val_'
        logs = {}

        # log stats before
        # if self.save_logs:
        #     loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
        #     self.logger.log_kv(f'{log_prefix}loss_before', loss_val)

        # train loop
        logs['loss'] = []
        logs['optim/lr'] = []
        logs['start_train_step'] = self.train_steps

        for ep in config_tqdm(range(epochs), suppress_fit_tqdm):
            for i_batch, sample_batched in enumerate(dataloader):

                if train:
                    self.optimizer.zero_grad()
                loss = self.loss(sample_batched)
                if isinstance(loss, tuple) and len(loss) == 2:
                    loss, loss_dict = loss
                elif isinstance(loss, tuple) and len(loss) != 2:
                    raise ValueError("Invalid loss")
                else:
                    loss_dict = {}

                if train:
                    loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.get_trainable_parameters(pretty_print=False), 
                            self.grad_clip_norm)
                    self.optimizer.step()
                    if self.scheduler and not self.scheduler_t_in_epochs:
                        self.scheduler.step()
                
                    optim_logs = {'epoch': self.train_epochs}
                    if len(loss_dict) > 0:
                        optim_logs.update(loss_dict)
                    if self.scheduler:
                        # We only have one parameter group for now.
                        optim_logs['optim/lr'] = self.get_current_learning_rate()
                    self.logger.log_loss(loss.item(), optim_logs=optim_logs)
                    self.train_steps += 1

                logs['loss'].append(loss.item())
                for k, v in loss_dict.items():
                    if logs.get(k) is None:
                        logs[k] = []
                    logs[k].append(v)
                if self.scheduler is not None:
                    logs['optim/lr'].append(self.get_current_learning_rate())
            
            if train:
                self.train_epochs += 1
                if self.scheduler and self.scheduler_t_in_epochs:
                    self.scheduler.step(self.train_epochs)
            else:
                self.val_epochs += 1

        logs['end_train_step'] = self.train_steps

        if train and reset_policy_after_epoch:
            params_after_opt = self.policy.policy.get_param_values()
            self.policy.policy.set_param_values(params_after_opt, set_new=True, set_old=True)

        # log stats after
        if train and self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            # loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            # self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))
        
        return logs

    def train(self, pixel=True, **kwargs):
        logs = self.fit(self.train_dl, epochs=1, train=True, **kwargs)
        log_dict = {
            'loss': np.mean(logs['loss'])
        }
        return log_dict
    
    def run_on_validation_data(self, pixel: bool = True, **kwargs):
        logs = self.fit(self.val_dl, epochs=1, train=False, **kwargs)
        log_dict = {
            'val_loss': np.mean(logs['loss']),
            'val_epoch': self.val_epochs
        }
        if self.loss_type == "MLE_multi_action":
            for k, v in logs.items():
                if k.startswith('loglik_action'):
                    log_dict[f'val_{k}'] = np.mean(v)
        self.logger.save_wb(self.train_steps, logs=log_dict)
        return log_dict

    def log_image_augmentation_data(self, num_batches: int, num_images_per_batch: int):
        import matplotlib.pyplot as plt

        path_to_save = Path(os.getcwd()) / 'train_imgs_after_aug'
        if not path_to_save.exists():
            path_to_save.mkdir()

        save_augs_and_return_cfg = {
            'use': True,
            # 'path': path_to_save / f'epoch_{self.epochs:03d}',
            'epoch': self.epochs,
            'num_images': num_images_per_batch,
        }
        for i_batch, batch in enumerate(self.train_dl):

            if i_batch >= num_batches:
                break

            save_augs_and_return_cfg['path'] = path_to_save / f'epoch_{self.epochs:03d}_iter_{i_batch:02d}'
            # Policy has the encoder for image observation
            out = self.policy.run_on_batch(batch, save_augs_and_return_cfg=save_augs_and_return_cfg)
            dir_to_save = path_to_save / f'epoch_{self.epochs:03d}_iter_{i_batch:02d}'
            if not dir_to_save.exists():
                dir_to_save.mkdir()
            for i_img in range(num_images_per_batch):
                static_img = out['static_imgs_after_aug'][i_img]
                hand_img = out['hand_imgs_after_aug'][i_img]
                fig = plt.figure(figsize=(16, 8))
                ax = fig.add_subplot(121)
                ax.imshow(static_img.transpose(1, 2, 0))
                ax = fig.add_subplot(122)
                ax.imshow(hand_img.transpose(1, 2, 0))
                task_name = out['tasks'][i_img]
                ax.set_title(task_name)
                plt.savefig(str(dir_to_save / f'task_{i_img:02d}_{task_name}.png'))
        print(f"Did save images in {path_to_save}")


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)
