import os, pickle
import itertools
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split

from mrest.utils.env_helpers import RLBenchEnvVariationData
from rlbench.demo import Demo


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
    demo: Optional[Demo]


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer) -> bool:
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    # HACK for tower3 task
    return episode_keypoints


def verify_waypoints_gripper_poses(demo_info: Demo, multi_traj_points: List[List[int]], key_frames: List[int]) -> bool:
    low_dim_data = np.copy(demo_info._observations[0].get_low_dim_data())
    robot_state_data_size = 37
    waypoint_idxs = [robot_state_data_size + x for x in [70, 77, 84, 98]]
    step_idx = 0
    for step, next_step in multi_traj_points:
        waypoint_pos = low_dim_data[waypoint_idxs[step_idx]:waypoint_idxs[step_idx] + 3]
        action = demo_info._observations[next_step].gripper_pose[:3]
        if not np.all(np.abs(waypoint_pos - action) < 1e-2):
            return False
        # assert np.all(np.abs(waypoint_pos - action) < 2e-2)
        step_idx += 1
        if step_idx >= len(waypoint_idxs):
            break
    
    return True


class RLBenchDataset(Dataset):
    def __init__(self,
                 job_data,
                 task_info: List[RLBenchEnvVariationData] = [],
                 camera_names: List = ['front_rgb'],
                 proprio_len: int = 4,
                 transform=None,
                 data_subsampling_cfg: Optional[Mapping[str, Any]] = None):
        self.transform = transform
        self.task_info = task_info
        self.camera_names = camera_names

        self.multi_temporal_sensors = job_data.get('multi_temporal_sensors')
        if self.multi_temporal_sensors is None or not self.multi_temporal_sensors.use:
            self.multi_temporal_sensors = None

        self.data_subsampling_cfg = data_subsampling_cfg

        self.num_tasks = len(self.task_info)

        self.proprio_alltasks = []
        self.actions_alltasks = []

        self.proprio_data_by_task = dict()
        self.image_names_by_task = dict()

        rlbench_data_cfg = job_data.rlbench_envs[job_data.rlbench_envs.use]
        self.rlbench_data_cfg = rlbench_data_cfg

        sequence_length = 1
        self.sequence_length = sequence_length

        self.datas = []
        self.task_names = []
        self.trajectory_data_indexes = []

        policy_type = job_data['policy_config']['type']
        use_tanh_action = job_data['policy_config'][policy_type]['policy_mlp_kwargs'].get('use_tanh_action', False)

        for task_index, task_info in enumerate(self.task_info):
            task_dir = Path(task_info.data_dir)
            task_name = task_info.name

            self.task_names.append(task_name)

            self.proprio_data_by_task[task_name] = dict()
            self.image_names_by_task[task_name] = dict()

            # data_demo_idxs = np.arange(start_idx,start_idx+num_demos_per_task)
            task_data_cfg = rlbench_data_cfg.data_dirs[task_info.env][task_info.data_key]
            data_demo_idxs = np.arange(0, task_data_cfg['num_train_demos'])

            for demo_idx, demo_dir in enumerate(task_dir.iterdir()):
                assert demo_dir.name.startswith('episode')
                if demo_idx in data_demo_idxs:
                    info_pickle = demo_dir / 'low_dim_obs.pkl'
                    with open(info_pickle, 'rb') as info_f:
                        demo_info = pickle.load(info_f)

                    # Discover keyframes
                    multi_traj_points = self.get_trajectory_points_for_training(demo_info)
                    key_frames = keypoint_discovery(demo_info)

                    # Read camera image paths
                    self.image_names_by_task[task_name][demo_dir] = dict()
                    for camera_name in camera_names:
                        img_dir = demo_dir / camera_name
                        demo_images = [f for f in img_dir.iterdir() if f.suffix == '.png']
                        demo_images = sorted(demo_images, key=lambda x: int(x.name.split('.')[0]))
                        self.image_names_by_task[task_name][demo_dir][camera_name] = demo_images

                    add_one_hot_timestep = False
                    if self.data_subsampling_cfg.get('add_one_hot_timestep') is not None:
                        add_one_hot_timestep = self.data_subsampling_cfg.add_one_hot_timestep.use

                    gripper_pose = []
                    gripper_open = []
                    ee_action = []
                    demo_data = {}
                    obs_low_dim_state = []

                    episode_length = len(multi_traj_points['step_next_step'])
                    for traj_point_idx, t in enumerate(multi_traj_points['step_next_step']):
                        step, next_step = t
                        o_t = demo_info._observations[step]
                        o_t_plus_one = demo_info._observations[next_step]

                        # Only for debugging purposes
                        # if not hasattr(o_t, 'action_norm'):
                        #     print(demo_dir)
                        #     continue

                        gripper_pose.append(o_t.gripper_pose)
                        gripper_open.append(np.array([o_t.gripper_open, 1 + o_t.gripper_open]))
                        obs_low_dim_state.append(np.copy(o_t.get_low_dim_data()))

                        if rlbench_data_cfg.get('use_action_from_pickle', False):
                            assert use_tanh_action, 'Should use tanh action by default for RLBench.'
                            if use_tanh_action:
                                # TODO(Mohit): Need to fix this correctly.
                                # ee_delta_xyz = o_t.action_norm
                                ee_delta_xyz = (o_t.action_norm * 2.) - 1.
                                ee_delta_gripper = o_t.gripper_open
                                # Using tanh actions
                                if ee_delta_gripper < 0.5:
                                    ee_delta_gripper = -1.0

                            else:
                                ee_delta_xyz = o_t.action_norm * (o_t.action_high - o_t.action_low) + o_t.action_low
                                ee_delta_gripper = o_t.gripper_open

                            ee_action.append(np.r_[ee_delta_xyz, ee_delta_gripper])
                        else:
                            assert not use_tanh_action

                            ee_delta_xyz = o_t_plus_one.gripper_pose[:3] - o_t.gripper_pose[:3]
                            ee_delta_gripper = o_t_plus_one.gripper_open
                            ee_action.append(np.r_[ee_delta_xyz, ee_delta_gripper])

                    demo_data['proprio'] = np.c_[np.stack(gripper_pose)[:, :3], np.stack(gripper_open)]
                    demo_data['actions'] = np.c_[ee_action]
                    demo_data['obs_low_dim_state'] = np.c_[obs_low_dim_state]
                    if self.data_subsampling_cfg and self.data_subsampling_cfg['add_low_dim_state']:
                        demo_data['proprio'] = demo_data['obs_low_dim_state']

                    self.proprio_data_by_task[task_name][demo_dir.name + f'_traj_0'] = demo_data

                    assert demo_data['actions'].shape[0] == demo_data['proprio'].shape[0]

                    curr_traj_start_data_idx = len(self.datas)
                    use_multitemporal_sensors = self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use
                    for traj_point_idx, traj_step in enumerate(multi_traj_points['step_next_step']):
                        step, next_step = traj_step

                        assert step == traj_point_idx

                        demo_image_path = dict()
                        for camera_name in camera_names:
                            cam_freq = 1 if not use_multitemporal_sensors else self.multi_temporal_sensors['rlbench'][camera_name]
                            last_sensor_step = (int(step // cam_freq)) * cam_freq
                            demo_image_path[camera_name] = self.image_names_by_task[task_name][demo_dir][camera_name][step]
                            assert step == int(demo_image_path[camera_name].name.split('.')[0]), (
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

    def get_trajectory_points_for_training(self, demo: Demo) -> Mapping[str, List[Tuple[int, int]]]:
        # Discover keyframes
        if self.data_subsampling_cfg is not None:
            use_subsampling = self.data_subsampling_cfg['use']
            subsampling_step = self.data_subsampling_cfg['subsampling_step']
            if self.data_subsampling_cfg.get('only_predict_waypoints') is not None:
                only_predict_waypoints = self.data_subsampling_cfg.only_predict_waypoints.use
            else:
                only_predict_waypoints = False

        else:
            use_subsampling = False
            only_predict_waypoints = False

        if not use_subsampling:
            steps = list(range(len(demo._observations)))
            return {
                'step_next_step':  [(x, y) for x, y in zip(steps[:-1], steps[1:])]
            }

        key_frames = keypoint_discovery(demo)
        if only_predict_waypoints:
            steps = [0] + [kf for kf in key_frames]
            return {
                'step_next_step':  [(x, y) for x, y in zip(steps[:-1], steps[1:])]
            }
        
        step_next_step = []
        for curr_wp, next_wp in zip(itertools.chain([0], key_frames[:-1]), itertools.chain(key_frames)):
            curr_step_next_step = [(x, y) for x, y in zip(range(curr_wp, next_wp, subsampling_step),
                                                    range(curr_wp + subsampling_step, next_wp, subsampling_step))]
            step_next_step.extend(curr_step_next_step)
        return {'step_next_step': step_next_step}

    def has_png_images(self):
        return True
    
    def get_task_data_key(self, task_info: RLBenchEnvVariationData, data_idx: int) -> str:
        return task_info.name + f'_data_{data_idx}'
    
    def get_task_name_from_task_data_key(self, task_data_key: str) -> str:
        return task_data_key.split('_data_')[0]

    def __len__(self):
        return len(self.datas)
    
    def _load_images_from_data_item(self, data_item: DataItem) -> Dict[str, np.ndarray]:
        images = dict()
        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use:
                # NOTE: For hand cameras last step will be first step
                image = np.asarray(Image.open(data_item.img_path[camera_name + '_last_step']))
            else:
                image = np.asarray(Image.open(data_item.img_path[camera_name]))
            images[camera_name] = torch.FloatTensor(image)
        return images
    
    def __getitem__(self, idx):
        data: DataItem = self.datas[idx]
        sample = self.get_low_dim_data(idx)
        sample.update(self._load_images_from_data_item(data))

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

        task_onehot = np.zeros((self.num_tasks))
        task_onehot[task_idx] = 1
        task_name = self.get_task_name_from_task_data_key(task_data_name)

        info = self.proprio_data_by_task[task_data_name][demo_name + f'_traj_{traj_idx}']
        proprio = info['proprio'][t]
        action = info['actions'][t]

        # action_no_noise = info['actions_no_noise'][t]
        # ee_xyz_no_noise = info['gripper_pose_no_noise'][t]

        sample = {
            'task': task_name,
            'task_enc': torch.FloatTensor(task_onehot),
            'proprio': torch.FloatTensor(proprio),
            'expert_actions': torch.FloatTensor(action),
            # 'expert_actions_no_noise': torch.FloatTensor(action_no_noise),
            # 'ee_xyz_no_noise': torch.FloatTensor(ee_xyz_no_noise),
        }

        return sample
