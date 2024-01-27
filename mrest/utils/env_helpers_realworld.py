from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import os
import gym
import numpy as np
import time
from collections import deque
from gym import spaces
from gym.spaces.box import Box
from pathlib import Path

from mrest.utils.gym_env import EnvSpec
from mrest.utils.env_helpers_realworld_constants import *
from pyquaternion import Quaternion


class RealWorldDictObs:
    def __init__(self,
                 task_name: str,
                 task_variation: int,
                 horizon: int,
                 task_descriptions: Optional[List[str]] = None,
                 proprio_key: Optional[str] = None,
                 tanh_action_cfg: Optional[Mapping[str, Any]] = None,
                 normalize_action_cfg: Optional[Mapping[str, Any]] = None,
                 multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
                 image_crop_cfg: Optional[Mapping[str, Any]] = None,
                 is_real_world: bool = False, 
                 *args, **kwargs) -> None:
        super().__init__()
        self.task_name = task_name

        self.width = 224
        self.height = 224
        self.is_real_world = is_real_world
        if self.is_real_world:
            from mrest.utils.env_helpers_realworld_franka import RealWorldFrankaEnv
            self.franka_env = RealWorldFrankaEnv(task_name)

        self.camera_names = ['static', 'hand']
        self.horizon = horizon

        self.image_crop_cfg = image_crop_cfg

        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors['use']:
            self.multi_temporal_sensors = None
        else:
            # TODO: Fix this logic, where we don't explicitly add multi_temporal_sensor frequency.
            self.last_img_obs_dict = dict()
            for cam_name in self.camera_names:
                sensor_freq = self.multi_temporal_sensors['realworld'].get(cam_name)
                assert sensor_freq is not None and sensor_freq > 0, f"FPS for camera not found: {cam_name}"
                self.last_img_obs_dict[cam_name] = deque(maxlen=1)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.tanh_action_cfg = tanh_action_cfg
        self.normalize_action_cfg = normalize_action_cfg
        if normalize_action_cfg is not None and normalize_action_cfg['use']:
            self.normalization_type = normalize_action_cfg['type']
            assert normalize_action_cfg['values_set'], "Values not set for normalization stats"

            self.mean_action = np.array(normalize_action_cfg['values']['mean'])
            self.std_action = np.array(normalize_action_cfg['values']['std'])

        else:
            self.normalize_action_cfg = None

        if tanh_action_cfg is not None and tanh_action_cfg.use:
            assert self.normalize_action_cfg is not None, "Normalization cannot be None"
            self.tanh_action_low = np.array(tanh_action_cfg['realworld']['low'])
            self.tanh_action_high = np.array(tanh_action_cfg['realworld']['high'])

        self.proprio_key = proprio_key

        self.multi_task = True
        assert task_descriptions is not None, 'Multi-task envs require some task description.'
        self.task_name = task_name
        self.task_variation = task_variation
        self.task_descriptions = task_descriptions
        self.multi_task_obs_dict = {
            'task': self.task_name + f'_var_{self.task_variation}',
            'task_descriptions': self.task_descriptions[:1],
        }

        self.action_size = 4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,))
        self._steps = 0

    @property
    def spec(self):
        return EnvSpec(1, self.action_size, self.horizon)

    def observation(self):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        obs_dict = {}
        if self.is_real_world:
            obs_dict = self.franka_env.get_observation()
            obs_dict = self._extract_obs(obs_dict)
        return obs_dict
    
    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        robot_state = obs['robot_state']
        force_torque = robot_state['O_F_ext_hat_K']
        # gripper_status = np.r_[robot_state['gripper_width'],
        #                        robot_state['gripper_is_grasped'],
        #                        1 + robot_state['gripper_is_grasped']]
        # gripper open hardcoded for now
        gripper_open = robot_state['gripper_open']
        gripper_status = np.r_[robot_state['gripper_joints'],
                               gripper_open,
                               1 + gripper_open ]

        q = np.array(robot_state['q'])
        gripper_tf = np.array(robot_state['O_T_EE']).reshape(4, 4).T
        gripper_position = gripper_tf[:3, 3]
        gripper_quat = Quaternion(matrix=gripper_tf[:3, :3], atol=1e-4)
        gripper_quat_arr = np.array([gripper_quat.w, gripper_quat.x, gripper_quat.y, gripper_quat.z])

        proprio = np.r_[
            gripper_position, gripper_quat_arr,
            # q,
            force_torque,
            gripper_status
        ]

        obs_dict = {
            'proprio': proprio,
            self.proprio_key: proprio,
        }
        # TODO: Should image numpy arrays be converted to tensors?
        for camera_name in self.camera_names:
            if camera_name == 'static' and obs[camera_name].shape[-1] == 4:
                image = obs[camera_name][:, :, :3]
            else:
                image = obs[camera_name]
            
            # Crop image (if required by config)
            if self.image_crop_cfg is not None and self.image_crop_cfg.use:
                crop_cfg = self.image_crop_cfg[camera_name]
                assert image.shape[0] == crop_cfg.org_size[0] and image.shape[1] == crop_cfg.org_size[1]
                start_uv, crop_size = crop_cfg.start_uv, crop_cfg.crop_size
                image = image[start_uv[0]:start_uv[0] + crop_size[0],
                              start_uv[1]:start_uv[1] + crop_size[1], :]
            
            obs_dict[camera_name] = image
        
        obs_dict.update(self.multi_task_obs_dict)
        self.current_EE_position = np.copy(gripper_position)

        return obs_dict

    def reset(self) -> Dict[str, np.ndarray]:
        # TODO: Wait for user input before continuing

        self._steps = 0
        if self.is_real_world:
            self.franka_env.reset()
            self.franka_env.stop_record_data()
            self.franka_env.start_record_data()
            # Wait to get an observation
            time.sleep(0.1)

            self.franka_env.start_motion_threaded()
        
        return self.observation()

    def unnormalize_action(self, action):
        if self.normalization_type == 'mean_std':
            return self.mean_std_unnormalize_action(action)
        elif self.normalization_type == 'tanh':
            return self.tanh_unnormalize_action(action, use_low_offset=False)
        elif self.normalization_type == 'tanh_fix':
            return self.tanh_unnormalize_action(action, use_low_offset=True)
            # return action
        else:
            return action
    
    def mean_std_unnormalize_action(self, action):
        unnormalized_action = self.mean_action + action * self.std_action
        return unnormalized_action

    def tanh_unnormalize_action(self, action, use_low_offset: bool = False):
        print(f"Unnormalizing action: {np.array_str(action, precision=4, suppress_small=True)}, use_low_offset: {use_low_offset}")
        action = np.clip(action, -1.0, 1.0)
        # Only scale position
        if len(self.tanh_action_high) == 3:
            if use_low_offset:
                action = (action + 1) / 2.0
                action[:3] = action[:3] * (self.tanh_action_high - self.tanh_action_low) + self.tanh_action_low
            else:
                # Bug in my normalization
                action[:3] = action[:3] * (self.tanh_action_high - self.tanh_action_low)
        else:
            if use_low_offset:
                action = (action + 1) / 2.0
                action = action * (self.tanh_action_high - self.tanh_action_low) + self.tanh_action_low
            else:
                # Bug in my normalization
                action = action * (self.tanh_action_high - self.tanh_action_low)
        return action

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        # Unnormalize action (if applicable)
        action = self.unnormalize_action(action)
        print(f"Action: {np.array_str(action, precision=4, suppress_small=True)}")

        # Clip actions to not go beylow certain z value
        ee_position = self.current_EE_position
        # Send this position to franka_env (for moving the robot)
        if self.is_real_world:
            # self.franka_env.do_one_step_motion(action)
            self.franka_env.do_continuous_motion(action)
        
        reward = 0
        terminate = False
        obs_dict = self.observation()
        self._steps += 1
        return obs_dict, reward, terminate, {}
    
    def close(self) -> None:
        if self.is_real_world:
            if self.franka_env.realtime_controller:
                self.franka_env.realtime_controller.join()
            self.franka_env.stop_record_data()
    
    def do_dynamic_episode_rollout(self, policy):
        assert self.is_real_world, "Dynamic rollouts only applicable in real world."

        self.franka_env.reset()
        self.franka_env.stop_record_data()

        # Wait to get an observation
        time.sleep(0.1)

        self.franka_env.start_record_data()

        def policy_cb():
            _obs = self.observation()
            inference_start_time = time.time()
            a, agent_info = policy.get_action(_obs)
            a = agent_info['evaluation']
            if not isinstance(a, np.ndarray):
                a = a.cpu().numpy().ravel()
            inference_end_time = time.time()
            print(f"Inference time: {inference_end_time - inference_start_time:.4f}")
            action = self.unnormalize_action(a)
            return action
        
        self.franka_env.start_dynamic_motion_with_policy_cb(policy_cb)


def create_realworld_env_with_name(job_data: Mapping[str, Any], 
                                   task_info,
                                   camera_names,
                                   is_real_world: bool = False):
    # task_name = 'RealworldBlockInsert'
    task_name = task_info.env
    task_variation = 0
    task_descriptions = REALWORLD_TASK_NAMES_BY_TASK_COMMAND[task_name]

    img_size = job_data.env_kwargs.image_width
    assert job_data.env_kwargs.image_width == job_data.env_kwargs.image_height

    realworld_env_type = job_data['realworld_envs']['use']
    realworld_env_cfg = job_data['realworld_envs'][realworld_env_type]

    env = RealWorldDictObs(
        task_name,
        task_variation,
        650,
        task_descriptions=task_descriptions,
        proprio_key='ee_xyz',
        tanh_action_cfg=job_data['env_kwargs'].get('tanh_action'),
        normalize_action_cfg=realworld_env_cfg.get('normalize_actions'),
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),
        is_real_world=is_real_world,
        image_crop_cfg=job_data['realworld_envs'].get('image_crop_cfg', None),
    )
    return env
    
