# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import gym
from gym.spaces.box import Box
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from collections import deque

from typing import Any, List, Mapping, Optional
from mrest.utils.logger import convert_config_dict_to_dict


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

    def get_image(self):
        if self.camera_name == "default":
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                  device_id=self.device_id)
        else:
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                  camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        return self.get_image()


class StepCountObservationWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.step_count = 0
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.step_count += 1
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError

    
class MuJoCoDictObs(StepCountObservationWrapper):
    def __init__(self, env, width, height, camera_names, device_id=-1,
                 depth=False, proprio_size: int = 0,
                 task_name: Optional[str] = None,
                 task_descriptions: Optional[List[str]] = None,
                 proprio_key: Optional[str] = None, 
                 multi_temporal_sensors: Optional[Mapping[str, Any]] = None, *args, **kwargs):
        super().__init__(env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_names = camera_names
        self.depth = depth
        self.proprio_size = proprio_size
        self.device_id = device_id
        if "v2" in env.spec.id:
            self.get_obs = env._get_obs

        self.proprio_key = proprio_key
        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors['use']:
            self.multi_temporal_sensors = None
        else:
            # TODO: Fix this logic, where we don't explicitly add multi_temporal_sensor frequency.
            self.multi_temporal_sensors = convert_config_dict_to_dict(multi_temporal_sensors)
            if self.multi_temporal_sensors.get('metaworld') is None:
                self.multi_temporal_sensors['metaworld'] = dict()
            self.last_img_obs_dict = dict()
            for cam_name in camera_names:
                if self.multi_temporal_sensors['metaworld'].get(cam_name) is None:
                    if 'hand' in cam_name:
                        self.multi_temporal_sensors['metaworld'][cam_name] = int(self.multi_temporal_sensors['ih_freq'])
                    else:
                        self.multi_temporal_sensors['metaworld'][cam_name] = int(self.multi_temporal_sensors['i3_freq'])

                sensor_freq = self.multi_temporal_sensors['metaworld'].get(cam_name)
                assert sensor_freq is not None and sensor_freq > 0, f"FPS for camera not found: {cam_name}"
                self.last_img_obs_dict[cam_name] = deque(maxlen=1)

        self.multi_task = False
        if kwargs.get('multi_task', False):
            self.multi_task = True
            assert task_descriptions is not None, 'Multi-task envs require some task description.'
            self.task_name = task_name
            self.task_descriptions = task_descriptions
            self.multi_task_obs_dict = {
                'task': self.task_name,
                'task_descriptions': self.task_descriptions,
            }

    def get_image(self):
        if self.camera_names == "default":
            img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                  device_id=self.device_id)
            img = img[::-1, :, :]
        else:
            try:
                img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                    camera_name='left_cap2', device_id=self.device_id)
                img = img[::-1, :, :]
            except:
                # For mujoco2.3 procedural env
                img = self.env.render(offscreen=True, camera_name='left_cap2', resolution=(self.width, self.height))
        return img

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.

        try:
            gt_state = self.env.unwrapped.get_obs()
        except:
            gt_state = self.env.unwrapped._get_obs()
        if self.proprio_key:
            proprio = gt_state[self.proprio_key]
        else:
            proprio = gt_state[:self.proprio_size]

        obs_dict = {
            'proprio': proprio,
            'gt_state': gt_state,
        }

        def _render_img(_camera_name: str):
            try:
                img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                                      camera_name=_camera_name, device_id=self.device_id)
                img = img[::-1, :, :]
            except:
                # For mujoco2.3 procedural env
                img = self.env.render(offscreen=True, camera_name=camera_name, resolution=(self.width, self.height))
            return img

        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors['use']:
                if self.step_count == 0:
                    img = _render_img(camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                elif self.step_count % self.multi_temporal_sensors['metaworld'][camera_name] == 0:
                    img = _render_img(camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                else:
                    assert len(self.last_img_obs_dict[camera_name]) == 1
                    img = self.last_img_obs_dict[camera_name][0]
                    new_img = False
                img = img[::-1, :, :]
            else:
                img = _render_img(camera_name)

            obs_dict.update({camera_name: img})

        if self.multi_task:
            obs_dict.update(self.multi_task_obs_dict)

        return obs_dict
