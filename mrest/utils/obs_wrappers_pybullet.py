# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import gym
from gym.spaces.box import Box
from omegaconf import OmegaConf

from collections import deque

from typing import Any, List, Mapping, Optional

class BallbotTSCStepCountObservationWrapper():

    def __init__(self, env, unnormalize_action, min_action, max_action, action_type, all_low_res, all_high_res):
        self.step_count = 0
        self.env = env
        self.unnormalize_action = unnormalize_action
        self.min_action = np.array([x for x in min_action])
        self.max_action = np.array([x for x in max_action])
        self.action_type = action_type
        self.all_low_res = all_low_res
        self.all_high_res = all_high_res

    def reset(self):
        o = self.env.reset()
        self.step_count = 0
        return self.observation()

    def step(self, action):
        # print('Normalized action:', action)
        if self.unnormalize_action:
            action = self.unnormalize(action)
            # print('UnNormalized action:', action)
        observation, reward, done, info = self.env.step_tsc(action, self.action_type, self.all_low_res, self.all_high_res)
        self.step_count += 1
        return self.observation(), reward, done, info

    def observation(self):
        raise NotImplementedError
    
    def unnormalize(self, action):
        action = (action + 1) / 2.0 # mapping [-1,1] to [0,1]
        # Only scale position
        if len(self.max_action) == 3:
            action[:3] = action[:3] * (self.max_action - self.min_action) + self.min_action
        else:
            action = action * (self.max_action - self.min_action) + self.min_action
        return action

    
class BallbotDictObs(BallbotTSCStepCountObservationWrapper):
    def __init__(self, env, width, height, camera_names, device_id=-1,
                 depth=False, proprio_size: int = 0,
                 task_name: Optional[str] = None,
                 task_descriptions: Optional[List[str]] = None,
                 proprio_key: Optional[str] = None, 
                 multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
                 min_action = None,
                 max_action = None,
                 action_type = 'delta_obs_pos',
                 all_low_res = False,
                 all_high_res = False,
                 *args, **kwargs):
        unnormalize_action = False if (min_action is None or max_action is None) else True
        super().__init__(env, unnormalize_action, min_action, max_action, action_type, all_low_res, all_high_res)
        self.observation_space = Box(low=0., high=255., shape=(4, width, height))
        self.width = width
        self.height = height
        self.camera_names = camera_names
        self.depth = depth
        self.proprio_size = proprio_size
        self.device_id = device_id
        # self.spec.observation_dim = 4
        # self.spec.action_dim = 4
        # self.horizon = env.spec.max_episode_steps

        self.proprio_key = proprio_key
        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors['use']:
            self.multi_temporal_sensors = None
        else:
            # TODO(saumya): Fix this logic, where we don't explicitly add multi_temporal_sensor frequency.
            self.multi_temporal_sensors = OmegaConf.to_container(multi_temporal_sensors)
            if self.multi_temporal_sensors.get('pybullet') is None:
                self.multi_temporal_sensors['pybullet'] = dict()
            self.last_img_obs_dict = dict()
            for cam_name in camera_names:
                if self.multi_temporal_sensors['pybullet'].get(cam_name) is None:
                    if 'hand' in cam_name:
                        self.multi_temporal_sensors['pybullet'][cam_name] = self.multi_temporal_sensors['ih_freq']
                    else:
                        self.multi_temporal_sensors['pybullet'][cam_name] = self.multi_temporal_sensors['i3_freq']

                sensor_freq = self.multi_temporal_sensors['pybullet'].get(cam_name)
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

    def observation(self):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.

        gt_state = self.env._get_obs()
        if self.proprio_key:
            proprio = gt_state[self.proprio_key]
        else:
            proprio = gt_state[:self.proprio_size]

        obs_dict = {
            'proprio': proprio,
            'gt_state': gt_state,
        }

        def _render_img(camera_name: str):
            return self.env.render(camera_name=camera_name, width=self.width, height=self.height)

        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors['use']:
                if self.step_count == 0:
                    img = _render_img(camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                elif self.step_count % self.multi_temporal_sensors['pybullet'][camera_name] == 0:
                    img = _render_img(camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                else:
                    assert len(self.last_img_obs_dict[camera_name]) == 1
                    img = self.last_img_obs_dict[camera_name][0]
                    new_img = False
            else:
                img = _render_img(camera_name)
            obs_dict.update({camera_name: img})

        if self.multi_task:
            obs_dict.update(self.multi_task_obs_dict)

        return obs_dict