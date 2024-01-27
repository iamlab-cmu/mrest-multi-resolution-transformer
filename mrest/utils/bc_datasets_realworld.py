import os, pickle
import itertools
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import default_collate

from mrest.utils.env_helpers import RealWorldEnvVariationData

from pyquaternion import Quaternion
from scipy.signal import savgol_filter


@dataclass
class DataItem:
    task: str
    task_index: int
    # Every trajectory can have multiple subsampled trajectories. This traj_index reprsents a particular
    # subsampling. In case we don't have subsampling this value will be 0.
    traj_index: int
    demo_dir: str
    img_path: Dict
    full_traj_step: int
    subsamp_traj_step: int
    demo: Optional[Mapping[str, Any]]


class RealWorldPegInsertDataset(Dataset):
    def __init__(self,
                 job_data,
                 task_info: List[RealWorldEnvVariationData] = [],
                 camera_names: List = ['static'],
                 proprio_len: int = 4,
                 transform=None,
                 data_subsampling_cfg: Optional[Mapping[str, Any]] = None,
                 image_crop_cfg: Optional[Mapping[str, Any]] = None,
                 is_val_data: bool = False):
        self.transform = transform
        self.task_info = task_info
        self.camera_names = camera_names
        self.is_val_data = is_val_data

        self.multi_temporal_sensors = job_data.get('multi_temporal_sensors')
        if self.multi_temporal_sensors is None or not self.multi_temporal_sensors.use:
            self.multi_temporal_sensors = None

        self.data_subsampling_cfg = data_subsampling_cfg
        self.image_crop_cfg = image_crop_cfg

        self.num_tasks = len(self.task_info)

        self.proprio_alltasks = []
        self.actions_alltasks = []

        self.proprio_data_by_task = dict()
        self.image_names_by_task = dict()

        realworld_data_cfg = job_data.realworld_envs[job_data.realworld_envs.use]
        self.realworld_data_cfg = realworld_data_cfg

        self.datas = []
        self.task_names = []
        self.trajectory_data_indexes = []

        policy_type = job_data['policy_config']['type']
        use_tanh_action = job_data['policy_config'][policy_type]['policy_mlp_kwargs'].get('use_tanh_action', False)
        self.use_tanh_action = use_tanh_action
        if use_tanh_action:
            assert job_data['env_kwargs']['tanh_action']['use']
            tanh_config = job_data['env_kwargs']['tanh_action']['realworld']
            self.tanh_config_low = np.array(tanh_config['low'])
            self.tanh_config_high = np.array(tanh_config['high'])

        task_index = 0
        for task_key, task_info in realworld_data_cfg['data_dirs'].items():
            for task_data_key, task_variation in task_info.items():
                task_dir = Path(task_variation.data)
                assert task_dir.exists(), f'Task dir does not exist: {task_dir}'

                # Get the task name
                task_name = f'{task_key}_var_0_data_{task_data_key}'
                self.task_names.append(task_name)

                self.proprio_data_by_task[task_name] = dict()
                self.image_names_by_task[task_name] = dict()

                # data_demo_idxs = np.arange(start_idx,start_idx+num_demos_per_task)
                task_data_cfg = task_variation
                if realworld_data_cfg.get('train_types') is not None:
                    assert len(realworld_data_cfg.train_types) > 0, "No train type specified."
                    data_demo_idxs = []
                    for train_type in realworld_data_cfg.train_types:
                        prefix = 'val' if is_val_data else 'train'
                        other_prefix = 'train' if is_val_data else 'val'
                        curr_demo_idxs = [x for x in task_data_cfg[f'{prefix}_demos_{train_type}']]
                        other_demo_idxs = [x for x in task_data_cfg[f'{other_prefix}_demos_{train_type}']]
                        # if len(set(curr_demo_idxs).intersection(set(other_demo_idxs))) != 0:
                        #     breakpoint()
                        assert len(set(curr_demo_idxs).intersection(set(other_demo_idxs))) == 0, (
                            "Train and validation demo indexes have overlap.")
                        
                        data_demo_idxs.extend(curr_demo_idxs)
                else:
                    prefix = 'val' if is_val_data else 'train'
                    other_prefix = 'train' if is_val_data else 'val'
                    curr_demo_idxs = [x for x in task_data_cfg[f'{prefix}_demos']]
                    other_demo_idxs = [x for x in task_data_cfg[f'{other_prefix}_demos']]
                    assert len(set(curr_demo_idxs).intersection(set(other_demo_idxs))) == 0, (
                        "Train and validation demo indexes have overlap.")

                all_demo_actions = []

                for _, demo_dir in enumerate(task_dir.iterdir()):
                    assert demo_dir.name.startswith('demo_')
                    demo_idx = int(demo_dir.name.split('demo_')[-1])
                    if demo_idx in data_demo_idxs:
                        info_pickle = demo_dir / 'info.pkl'
                        with open(info_pickle, 'rb') as info_f:
                            demo_info = pickle.load(info_f)
                        
                        multi_traj_points = self.get_trajectory_points_for_training(demo_info)

                        # Read camera image paths
                        self.image_names_by_task[task_name][demo_dir] = dict()
                        for camera_name in camera_names:
                            img_dir = demo_dir / camera_name
                            demo_images = [f for f in img_dir.iterdir() if f.suffix == '.png']
                            demo_images = sorted(demo_images, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                            self.image_names_by_task[task_name][demo_dir][camera_name] = demo_images

                        gripper_poses = []
                        # Used for state noise injection
                        gripper_poses_next = []
                        gripper_open = []
                        robot_joints = []
                        robot_force_torque = []
                        ee_action = []
                        demo_data = {}
                        obs_low_dim_state = []

                        def _get_gripper_pose_from_rigid_transform(_pose_tf) -> np.ndarray:
                            _pose_tf = np.array(_pose_tf).reshape(4, 4).T
                            _gripper_position = _pose_tf[:3, 3]
                            try:
                                _gripper_quat = Quaternion(matrix=_pose_tf[:3, :3], atol=1e-4)
                                _gripper_quat_arr = np.array([_gripper_quat.w, _gripper_quat.x, _gripper_quat.y, _gripper_quat.z])
                            except ValueError as e:
                                _gripper_quat = Quaternion(matrix=_pose_tf[:3, :3], atol=1e-3)
                                _gripper_quat_arr = np.array([_gripper_quat.w, _gripper_quat.x, _gripper_quat.y, _gripper_quat.z])

                            return np.r_[_gripper_position, _gripper_quat_arr]

                        for traj_point_idx, t in enumerate(multi_traj_points['step_next_step']):
                            step, next_step = t

                            gripper_pose_curr = _get_gripper_pose_from_rigid_transform(demo_info['O_T_EE'][step])
                            gripper_pose_next = _get_gripper_pose_from_rigid_transform(demo_info['O_T_EE'][next_step])
                            q = np.array(demo_info['q'][step])
                            # dq = np.array(demo_info['dq'][step])
                            force_torque = np.array(demo_info['O_F_ext_hat_K'][step])
                            # Old code
                            # gripper_status = np.r_[demo_info['gripper_width'][step],
                            #                        demo_info['gripper_is_grasped'][step],
                            #                        1 + demo_info['gripper_is_grasped'][step]]
                            gripper_status = np.r_[demo_info['gripper_joints'][step],
                                                   demo_info['gripper_open'][step],
                                                   1 + demo_info['gripper_open'][step] ]
                            
                            if data_subsampling_cfg.get('proprio_history') is not None and data_subsampling_cfg.proprio_history.use:
                                hist_len = data_subsampling_cfg.proprio_history.size
                                gripper_pose_curr_with_hist = []
                                for hist_idx in range(hist_len):
                                    if (step - hist_idx) < 0:
                                        gripper_pose_curr_with_hist.append(np.zeros(7,))
                                    else:
                                        gripper_pose_curr_with_hist.append(
                                            _get_gripper_pose_from_rigid_transform(demo_info['O_T_EE'][step - hist_idx]))
                                gripper_poses.append(np.hstack(gripper_pose_curr_with_hist))
                            else:
                                gripper_poses.append(gripper_pose_curr)

                            gripper_poses_next.append(gripper_pose_next)
                            robot_joints.append(q)
                            robot_force_torque.append(force_torque)
                            gripper_open.append(gripper_status)
                            
                            # TODO: Write an extra wrapper for longer trajectories to use this.
                            ee_delta_xyz = demo_info['O_T_EE'][next_step][-4:-1] - demo_info['O_T_EE'][step][-4:-1]
                            # ee_delta_gripper = 0 # Old code
                            if self.data_subsampling_cfg.get('subsample_after_smoothening', False):
                                subsample_step = self.data_subsampling_cfg.subsampling_step
                                if step + subsample_step >= len(demo_info['O_T_EE']):
                                    ee_delta_gripper = demo_info['gripper_open'][-1]
                                else:
                                    ee_delta_gripper = demo_info['gripper_open'][step + subsample_step]
                            else:
                                ee_delta_gripper = demo_info['gripper_open'][next_step]

                            ee_action.append(np.r_[ee_delta_xyz, ee_delta_gripper])
                    
                        use_joints_in_proprio = realworld_data_cfg.get('use_joints_in_proprio', True)
                        if use_joints_in_proprio:
                            demo_data['proprio'] = np.c_[np.stack(gripper_poses), np.stack(robot_joints),
                                                         np.stack(robot_force_torque), np.stack(gripper_open)]
                        else:
                            demo_data['proprio'] = np.c_[np.stack(gripper_poses), np.stack(robot_force_torque),
                                                         np.stack(gripper_open),]
                        demo_data['actions'] = np.c_[ee_action]
                        demo_data['gripper_pose_next'] = np.stack(gripper_poses_next)

                        if self.data_subsampling_cfg.get('subsample_after_smoothening', False):
                            # NOTE: Here we smooth out the gripper positions and then from smooth gripper positions
                            # we derive our delta actions.
                            assert self.data_subsampling_cfg.savgol_filter.use
                            gripper_positions_non_smooth = np.stack(gripper_poses)[:, :3]
                            gripper_positions_smooth = []
                            for i in range(3):
                                if task_data_cfg.get('subsample_after_smoothening_cfg') is not None:
                                    window_traj_indexes = task_data_cfg.subsample_after_smoothening_cfg.savgol_window_traj_idx
                                    filter_window_lengths = task_data_cfg.subsample_after_smoothening_cfg.savgol_window_lengths
                                    filtered_windows = []
                                    for window_idx, window_start in enumerate(window_traj_indexes):
                                        if window_idx == len(window_traj_indexes) - 1:
                                            window_end = -1
                                        else:
                                            window_end = window_start + window_traj_indexes[window_idx + 1]
                                        filtered_window = savgol_filter(
                                            gripper_positions_non_smooth[window_start:window_end, i],
                                            filter_window_lengths[window_idx],
                                            self.data_subsampling_cfg.savgol_filter.polyorder,)
                                        filtered_windows.append(filtered_window)
                                    gripper_positions_smooth.append(np.concatenate(filtered_windows))

                                else:
                                    gripper_positions_smooth.append(savgol_filter(
                                        # gripper_positions_non_smooth[:, i] - gripper_positions_non_smooth[0, i],
                                        gripper_positions_non_smooth[:, i],
                                        self.data_subsampling_cfg.savgol_filter.window_length,
                                        self.data_subsampling_cfg.savgol_filter.polyorder,))

                            gripper_positions_smooth = np.c_[gripper_positions_smooth].T
                            subsample_step = self.data_subsampling_cfg.subsampling_step

                            data_subsampling_cfg.filter_out_zero_action_demo_steps

                            if (data_subsampling_cfg.filter_out_zero_action_demo_steps and
                                data_subsampling_cfg.get('actions_future') is not None and
                                data_subsampling_cfg['actions_future'].use):
                                # PASS 
                                assert subsample_step == max(data_subsampling_cfg['actions_future']['steps'])
                                smooth_step_next_step = self.filter_ee_positions_with_no_motion(
                                    gripper_positions_smooth, subsample_step)
                                smooth_step_next_step_arr = np.array(smooth_step_next_step, dtype=np.int32)
                                all_ee_actions, all_ee_actions_noisy = [], []
                                for action_future_step in data_subsampling_cfg['actions_future']['steps']:
                                    curr_step = smooth_step_next_step_arr[:, 0]
                                    next_step = curr_step + action_future_step
                                    # Noiseless actions
                                    ee_actions = (
                                        gripper_positions_smooth[next_step, :] - 
                                        gripper_positions_smooth[curr_step, :]
                                    )
                                    all_ee_actions.append(ee_actions)
                                    # Noisy actions
                                    ee_actions_noisy = (
                                        gripper_positions_non_smooth[next_step, :] - 
                                        gripper_positions_non_smooth[curr_step, :]
                                    )
                                    all_ee_actions_noisy.append(ee_actions_noisy)

                                all_ee_actions.append(demo_data['actions'][smooth_step_next_step_arr[:, 0], -1:])
                                # Stack all actions together
                                demo_data['actions'] = np.hstack(all_ee_actions)
                                demo_data['proprio'] = demo_data['proprio'][smooth_step_next_step_arr[:, 0], :]
                                # Update (step, next_step)
                                multi_traj_points['step_next_step'] = smooth_step_next_step

                                # Store noisy non-smoothed actions as well
                                demo_data['actions_noisy'] = np.hstack(all_ee_actions_noisy)

                            # We should now remove out parts of smooth trajectory that did not have any motion.
                            # Remove 0 action steps.
                            # NOTE: When we smooth out actions we directly remove 0 action steps initially itself.
                            elif data_subsampling_cfg.filter_out_zero_action_demo_steps:
                                smooth_step_next_step = self.filter_ee_positions_with_no_motion(gripper_positions_smooth, subsample_step)
                                smooth_step_next_step_arr = np.array(smooth_step_next_step, dtype=np.int32)
                                # Create actions from the smooth_steps but without using zero-action steps.
                                ee_actions = (
                                    gripper_positions_smooth[smooth_step_next_step_arr[:, 1], :] - 
                                    gripper_positions_smooth[smooth_step_next_step_arr[:, 0], :]
                                )
                                demo_data['actions'] = np.c_[
                                    ee_actions, 
                                    demo_data['actions'][smooth_step_next_step_arr[:, 0], -1],
                                ]
                                # Ensure proprio values are aligned with the actions.
                                demo_data['proprio'] = demo_data['proprio'][smooth_step_next_step_arr[:, 0], :]
                                # Update (step, next_step)
                                multi_traj_points['step_next_step'] = smooth_step_next_step

                                # Store noisy non-smoothed actions as well
                                demo_data['actions_noisy'] = (
                                    gripper_positions_non_smooth[smooth_step_next_step_arr[:, 1], :] - 
                                    gripper_positions_non_smooth[smooth_step_next_step_arr[:, 0], :]
                                )
                            

                            else:
                                ee_actions = gripper_positions_smooth[subsample_step:, :] - gripper_positions_smooth[:-subsample_step, :]
                                ee_actions = np.vstack([np.zeros((2, 3)), ee_actions, np.zeros((subsample_step - 2, 3))])
                                demo_data['actions'] = np.c_[ee_actions, demo_data['actions'][:, -1]]

                                # Store noisy non-smoothed actions as well
                                demo_data['actions_noisy'] = (
                                    gripper_positions_non_smooth[subsample_step:, :] - 
                                    gripper_positions_non_smooth[:-subsample_step, :]
                                )

                        elif (self.data_subsampling_cfg.get('savgol_filter', None) is not None and
                            # Smoothen out the data. 
                            # NOTE: Here we directly smooth out the actions.
                            self.data_subsampling_cfg.savgol_filter.use):
                            demo_data['actions_noisy'] = np.copy(demo_data['actions'])
                            for action_idx in range(3):
                                demo_data['actions'][:, action_idx] = savgol_filter(
                                    demo_data['actions'][:, action_idx],
                                    self.data_subsampling_cfg.savgol_filter.window_length,
                                    self.data_subsampling_cfg.savgol_filter.polyorder)
                        else:
                            print("**Warning not smoothing any actions**")
                            raise NotImplementedError

                        self.proprio_data_by_task[task_name][demo_dir.name + f'_traj_0'] = demo_data

                        assert demo_data['actions'].shape[0] == demo_data['proprio'].shape[0]

                        curr_traj_start_data_idx = len(self.datas)
                        use_multitemporal_sensors = self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use
                        for traj_point_idx, traj_step in enumerate(multi_traj_points['step_next_step']):
                            step, _ = traj_step

                            demo_image_path = dict()
                            for camera_name in camera_names:
                                cam_freq = 1 if not use_multitemporal_sensors else self.multi_temporal_sensors['realworld'][camera_name]
                                if cam_freq == 1:
                                    last_sensor_step = step
                                elif cam_freq > 1:
                                    last_sensor_step = (int(step // cam_freq)) * cam_freq
                                demo_image_path[camera_name] = self.image_names_by_task[task_name][demo_dir][camera_name][step]
                                assert step == int(demo_image_path[camera_name].name.split('.')[0].split('_')[1]), (
                                    'Image and step do not match')
                                demo_image_path[camera_name + '_last_step'] = (
                                    self.image_names_by_task[task_name][demo_dir][camera_name][last_sensor_step])

                            di = DataItem(task_name, task_index, 0, demo_dir.name, demo_image_path, step, traj_point_idx, demo_info)
                            self.datas.append(di)
                        curr_traj_end_data_idx = len(self.datas)
                        self.trajectory_data_indexes.append((curr_traj_start_data_idx, curr_traj_end_data_idx))

                # if demo_idx >= num_demos_per_task:
                #     break

        self._has_png_images = self.has_png_images()
    
    def get_action_stats_for_dataset(self):
        all_actions = []
        for task_name, task_demo_data in self.proprio_data_by_task.items():
            for demo_key, demo_data in task_demo_data.items():
                all_actions.append(demo_data['actions'])
        all_actions_arr = np.vstack(all_actions)
        eps = 1e-8
        stats = {
            'mean': all_actions_arr.mean(axis=0),
            'std': all_actions_arr.std(axis=0) + eps,
            'min': all_actions_arr.min(axis=0),
            'max': all_actions_arr.max(axis=0),
        }
        return stats
    
    def filter_ee_positions_with_no_motion(self, ee_position: np.ndarray, subsampling_step: int, 
                                           future_steps: Optional[List[int]] = None):
        """Remove steps where for subsampling_steps ee_position does not change."""
        step_next_step = []
        for curr_step, curr_ee_position in enumerate(ee_position[:-subsampling_step]):
            has_motion = False
            for next_step in range(curr_step + 1, curr_step + subsampling_step):
                delta_xyz = np.abs(ee_position[next_step] - curr_ee_position)
                has_motion |= (np.any(delta_xyz > 0.002))
            if has_motion or curr_step < 5:
                step_next_step.append((curr_step, curr_step + subsampling_step))
        return step_next_step
    
    def filter_demos_with_no_motion(self, demo_info, subsampling_step: int):
        """Remove steps where for subsampling_steps we perform no motion."""
        ee_positions = [curr_ee_pose[-4:-1] for curr_ee_pose in demo_info['O_T_EE']]
        return self.filter_ee_positions_with_no_motion(ee_positions, subsampling_step)
            
    def get_trajectory_points_for_training(self, demo_info: Mapping[str, Any]) -> List[List[Any]]:
        if self.data_subsampling_cfg is not None:
            use_subsampling = self.data_subsampling_cfg['use']
            subsampling_step = self.data_subsampling_cfg['subsampling_step']

            # We subsample actions after smoothening per step actions
            if self.data_subsampling_cfg.get('subsample_after_smoothening', False):
                use_subsampling = False
        else:
            use_subsampling = False
        
        steps = demo_info['step_idx']

        if use_subsampling:
            step_next_step = [(x, y) for x, y in zip(steps[:-subsampling_step], steps[subsampling_step:])]
            step_next_step.extend([(x, steps[-1]) for x in steps[-subsampling_step + 1:-1]])

            if self.data_subsampling_cfg.get('filter_out_zero_action_demo_steps', False):
                filter_type = self.data_subsampling_cfg.get('filter_type', 'step')
                if filter_type == 'step':
                    filter_step_next_step = []
                    for step, next_step in step_next_step:
                        ee_delta_xyz = demo_info['O_T_EE'][next_step][-4:-1] - demo_info['O_T_EE'][step][-4:-1]
                        if np.all(np.abs(ee_delta_xyz) < 0.001):
                            pass
                        else:
                            filter_step_next_step.append((step, next_step))
                    step_next_step = filter_step_next_step
                elif filter_type == 'traj':
                    _, step_next_step = self.filter_demos_with_no_motion(demo_info, subsampling_step)
                else:
                    raise ValueError(f"Invalid filter type: {filter_type}")
            
            return {
                'step_next_step':  step_next_step,
            }
        else:
            return {
                'step_next_step':  [(x, y) for x, y in zip(steps[:-1], steps[1:])]
            }

    def has_png_images(self):
        return True
    
    def get_task_data_key(self, task_info: RealWorldEnvVariationData, data_idx: int) -> str:
        return task_info.name + f'_data_{data_idx}'
    
    def get_task_name_from_task_data_key(self, task_data_key: str) -> str:
        return task_data_key.split('_data_')[0]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data: DataItem = self.datas[idx]
        sample = self.get_low_dim_data(idx)

        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use:
                image = np.asarray(Image.open(data.img_path[camera_name + '_last_step']))
            else:
                image = np.asarray(Image.open(data.img_path[camera_name]))
            
            # Crop image
            if self.image_crop_cfg is not None and self.image_crop_cfg.use:
                crop_cfg = self.image_crop_cfg[camera_name]
                assert image.shape[0] == crop_cfg.org_size[0] and image.shape[1] == crop_cfg.org_size[1]
                start_uv, crop_size = crop_cfg.start_uv, crop_cfg.crop_size
                image = image[start_uv[0]:start_uv[0] + crop_size[0],
                              start_uv[1]:start_uv[1] + crop_size[1], :]

            if camera_name == 'static' and image.shape[-1] == 4:
                sample.update({camera_name: torch.FloatTensor(image[:, :, :3])})
            else:
                sample.update({camera_name: torch.FloatTensor(image)})

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_low_dim_data(self, idx):
        data: DataItem = self.datas[idx]
        task_data_name = data.task
        task_idx = data.task_index
        traj_idx = data.traj_index
        demo_name = data.demo_dir
        # t = data.step
        t = data.subsamp_traj_step

        task_onehot = np.zeros((0,))
        task_name = self.get_task_name_from_task_data_key(task_data_name)

        info = self.proprio_data_by_task[task_data_name][demo_name + f'_traj_{traj_idx}']
        proprio = info['proprio'][t]
        action = info['actions'][t]
        if 'actions_noisy' in info:
            action_noisy = info['actions_noisy'][t]
        else:
            action_noisy = None

        # Add noise to actions.
        if self.realworld_data_cfg.get('state_noise', None) is not None and self.realworld_data_cfg.state_noise.use:
            proprio_noise_min = np.array(self.realworld_data_cfg.state_noise.proprio_noise_min)
            proprio_noise_max = np.array(self.realworld_data_cfg.state_noise.proprio_noise_max)
            ee_start_noise = np.random.uniform(proprio_noise_min, proprio_noise_max)
            proprio[:3] = proprio[:3] + ee_start_noise
            action[:3] = action[:3] - ee_start_noise

        if self.realworld_data_cfg.normalize_actions['use']:
            norm_type = self.realworld_data_cfg.normalize_actions.get('type', 'mean_std')
            if norm_type == 'mean_std':
                action_mean = np.array(self.realworld_data_cfg.normalize_actions['values']['mean'])
                action_std = np.array(self.realworld_data_cfg.normalize_actions['values']['std'])
                action = (action - action_mean) / action_std
            elif norm_type == 'tanh':
                assert self.use_tanh_action
                norm_action = (action) / (self.tanh_config_high - self.tanh_config_low)
                action = np.clip(norm_action, -1, 1)
            elif norm_type == 'tanh_fix':
                assert self.use_tanh_action
                if 'actions_future' in self.data_subsampling_cfg and self.data_subsampling_cfg.actions_future.use:
                    action_len = len(self.data_subsampling_cfg.actions_future.steps)
                    tanh_config_low = np.repeat(self.tanh_config_low, action_len)[:-(action_len - 1)]
                    tanh_config_high = np.repeat(self.tanh_config_high, action_len)[:-(action_len - 1)]
                else:
                    tanh_config_low, tanh_config_high = self.tanh_config_low, self.tanh_config_high

                norm_action = (action - tanh_config_low) / (tanh_config_high - tanh_config_low)
                action = np.clip(norm_action * 2. - 1., -1, 1)
            else:
                raise ValueError(f'Unknown norm_type: {norm_type}')

        # action_no_noise = info['actions_no_noise'][t]
        # ee_xyz_no_noise = info['gripper_pose_no_noise'][t]

        sample = {
            'task': task_name,
            'task_enc': torch.FloatTensor(task_onehot),
            'proprio': torch.FloatTensor(proprio),
            'expert_actions': torch.FloatTensor(action),
            # 'expert_actions_no_noise': torch.FloatTensor(action_no_noise),
            # 'ee_xyz_no_noise': torch.FloatTensor(ee_xyz_no_noise),
            'action_noisy': torch.FloatTensor(action_noisy) if action_noisy is not None else None,
        }

        return sample
