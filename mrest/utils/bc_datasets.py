import os, pickle
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, random_split


from mrest.utils.env_helpers import get_metaworld_target_object_name


@dataclass
class DataItem:
    task: str
    task_index: int
    demo_dir: str
    img_path: Dict
    step: int
    sequence_indices: Tuple[int, int, int, int]


class BCMultiTaskVariableLenImageProprioDataset(Dataset):
    def __init__(self,
                 data_dir,
                 task_names=['assembly-v2-goal-observable'],
                 num_demos_per_task=15,
                 num_demos_train_per_task_cfg: Optional[Mapping[str, Any]] = None,
                 start_idx=0,
                 camera_names: List = ['left_cap2'],
                 proprio_len: int = 4,
                 append_object_mask=None,
                 transform=None,
                 multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
                 env_config_dict: Optional[Mapping[str, Any]] = None,
                 job_data: Mapping[str, Any] = {},):

        _data_folders = [f for f in Path(data_dir).iterdir()]
        self.is_multiview_dataset = False if 'left_cap2' in _data_folders else True

        if not self.is_multiview_dataset:
            data_dir = os.path.join(data_dir, camera_names[0])

        self.data_dir_path = Path(data_dir)
        self.transform = transform

        self.num_demos_per_task = num_demos_per_task
        self.metaworld_env_cfg = None
        if job_data is not None:
            self.metaworld_env_cfg = job_data['metaworld_envs'][job_data['metaworld_envs']['use']]
            self.num_demos_train_per_task_cfg = num_demos_train_per_task_cfg

        self.task_names = task_names
        self.camera_names = camera_names
        self.append_object_mask = append_object_mask
        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors.use:
            self.multi_temporal_sensors = None
        # This is the env_configs loaded from pickle file.
        self.env_config_dict = env_config_dict

        self.num_tasks = len(self.task_names)
        self.dataset_start_idx = start_idx

        self.proprio_alltasks = []
        self.actions_alltasks = []

        self.proprio_data_by_task = dict()
        self.image_names_by_task = dict()

        self.datas = []

        for task_index, task_name in enumerate(self.task_names):
            task_dir = self.data_dir_path / task_name

            self.proprio_data_by_task[task_name] = dict()
            self.image_names_by_task[task_name] = dict()

            if (self.metaworld_env_cfg is not None and 
                self.metaworld_env_cfg.get('procedural_objects', False) and
                self.num_demos_train_per_task_cfg is not None):

                assert self.env_config_dict is not None
                target_object_name = get_metaworld_target_object_name(self.env_config_dict[task_name])
                skill_name = self.env_config_dict[task_name]['skill']
                if (target_object_name in self.num_demos_train_per_task_cfg['target_obj_skill_demo_counts'] and
                    skill_name in self.num_demos_train_per_task_cfg['target_obj_skill_demo_counts'][target_object_name]):
                    num_demos_per_current_task = self.num_demos_train_per_task_cfg['target_obj_skill_demo_counts'][
                        target_object_name][skill_name]
                    print(f"Task: {task_name}, target: {target_object_name}, skill: {skill_name}, num_demos: {num_demos_per_current_task}")

                else:
                    num_demos_per_current_task = num_demos_per_task
            else:
                num_demos_per_current_task = num_demos_per_task

            data_demo_idxs = np.arange(self.dataset_start_idx, self.dataset_start_idx + num_demos_per_current_task)

            for demo_idx, demo_dir in enumerate(task_dir.iterdir()):
                assert demo_dir.name.startswith('demo')
                if demo_idx in data_demo_idxs:
                    info_pickle = demo_dir / 'info.pickle'
                    with open(info_pickle, 'rb') as info_f:
                        demo_info = pickle.load(info_f)
                    demo_info['proprio'] = demo_info['observations'][:, :proprio_len]
                    del demo_info['observations']
                    self.proprio_data_by_task[task_name][demo_dir.name] = demo_info
                    self.image_names_by_task[task_name][demo_dir] = dict()

                    demo_images = []
                    for camera_name in camera_names:
                        if not self.is_multiview_dataset:
                            demo_images = [f for f in demo_dir.iterdir() if f.suffix == '.png']
                        else:
                            img_dir = demo_dir / camera_name
                            demo_images = [f for f in img_dir.iterdir() if f.suffix == '.png']
                        # images have names 'img_t{}.png'
                        demo_images = sorted(demo_images, key=lambda x: int(x.name.split('img_t')[1].split('.')[0]))
                        self.image_names_by_task[task_name][demo_dir][camera_name] = demo_images

                    assert len(demo_images) > 0, "No image found"

                    use_multitemporal_sensors = self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use
                    if use_multitemporal_sensors:
                        static_cam_freq = self.multi_temporal_sensors.i3_freq
                        hand_cam_freq = self.multi_temporal_sensors.ih_freq
                    else:
                        static_cam_freq = 1
                        hand_cam_freq = 1
                    
                    print(f"Demo dir: {demo_dir}, num_images: {len(demo_images)}")
                    for step in range(len(demo_images)):
                        demo_image_path = dict()
                        for camera_name in camera_names:
                            cam_freq = static_cam_freq if self.is_static_camera(camera_name) else hand_cam_freq
                            last_sensor_step = (int(step // cam_freq)) * cam_freq
                            demo_image_path[camera_name] = self.image_names_by_task[task_name][demo_dir][camera_name][step]
                            assert step == int(demo_image_path[camera_name].name.split('img_t')[1].split('.')[0]), (
                                'Image and step do not match')
                            demo_image_path[camera_name + '_last_step'] = (
                                self.image_names_by_task[task_name][demo_dir][camera_name][last_sensor_step])

                        if self.append_object_mask == 'mdetr':
                            demo_image_path['object_mask'] = demo_dir / 'heatmap' / 'img_t0_mdetr.png'
                        elif self.append_object_mask == 'owl_vit':
                            demo_image_path['object_mask'] = demo_dir / 'heatmap' / 'img_t0_owlvit.png'

                        di = DataItem(task_name, task_index, demo_dir.name, demo_image_path, step, None)
                        self.datas.append(di)
                    
                # if demo_idx >= num_demos_per_task:
                #     break

        self._has_png_images = True
        # Sometimes validation dataset is 0
        if len(self.datas):
            img_data = self.load_images_from_data(self.datas[0])
            img_shapes = [v.shape for v in img_data.values()]
            self.img_shape = (img_shapes[0])

    def is_static_camera(self, cam_name):
        if 'hand' in cam_name:
            return False
        assert cam_name == 'left_cap2', "Only one camera for now"
        return True
    
    def __len__(self):
        return len(self.datas)
    
    def load_images_from_data(self, data: DataItem):
        sample = {}
        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use:
                # NOTE: For hand cameras last step will be first step
                image = np.asarray(Image.open(data.img_path[camera_name + '_last_step']))
            else:
                image = np.asarray(Image.open(data.img_path[camera_name]))

            if self.append_object_mask is not None and camera_name == 'left_cap2':
                object_mask = np.asarray(Image.open(data.img_path['object_mask']))
                image = np.append(image, object_mask[:, :, None], axis=2)
                # sample.update({'object_mask': torch.FloatTensor(object_mask)})

            sample[camera_name] = torch.FloatTensor(image)
        return sample

    def __getitem__(self, idx):
        data: DataItem = self.datas[idx]
        task_name = data.task
        task_idx = data.task_index
        demo_name = data.demo_dir
        t = data.step

        task_onehot = np.zeros((self.num_tasks))
        task_onehot[task_idx] = 1
        task_name = self.task_names[task_idx]

        info = self.proprio_data_by_task[task_name][demo_name]
        proprio = info['proprio'][t]
        action = info['actions'][t]

        sample = {
            'task': task_name,
            'task_enc': torch.FloatTensor(task_onehot),
            'proprio': torch.FloatTensor(proprio),
            'expert_actions': torch.FloatTensor(action),
        }

        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use:
                # NOTE: For hand cameras last step will be first step
                # TODO(saumya): remove hardcoded [:,:,:3]
                image = np.asarray(Image.open(data.img_path[camera_name + '_last_step']))[:,:,:3]
            else:
                # TODO(saumya): remove hardcoded [:,:,:3]
                image = np.asarray(Image.open(data.img_path[camera_name]))[:,:,:3]

            if self.append_object_mask is not None and camera_name == 'left_cap2':
                object_mask = np.asarray(Image.open(data.img_path['object_mask']))
                image = np.append(image, object_mask[:,:,None], axis=2)
                # sample.update({'object_mask': torch.FloatTensor(object_mask)})

            sample.update({camera_name: torch.FloatTensor(image)})

        if self.transform:
            sample = self.transform(sample)
        return sample


class BCPybulletMultiTaskVariableLenImageProprioDataset(Dataset):
    def __init__(self,
                data_dir,
                task_names=['assembly-v2-goal-observable'],
                num_demos_per_task=15,
                start_idx=0,
                camera_names: List = ['left_cap2'],
                proprio_len: int = 4,
                append_object_mask=None,
                transform=None,
                multi_temporal_sensors: Optional[Mapping[str, Any]] = None, 
                min_action = None,
                max_action = None,
                action_type = 'delta_obs_pos'):

        self.normalize_action = False if (min_action is None or max_action is None) else True
        self.action_type = action_type
        if self.normalize_action:
            self.min_action = np.array(min_action)
            self.max_action = np.array(max_action)

        _data_folders = [f for f in Path(data_dir).iterdir()]
        self.is_multiview_dataset = False if 'left_cap2' in _data_folders else True

        if not self.is_multiview_dataset:
            data_dir = os.path.join(data_dir, camera_names[0])

        self.data_dir_path = Path(data_dir)
        self.transform = transform
        self.num_demos_per_task = num_demos_per_task
        self.task_names = task_names
        self.camera_names = camera_names
        self.append_object_mask = append_object_mask
        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors.use:
            self.multi_temporal_sensors = None

        self.num_tasks = len(self.task_names)

        self.proprio_alltasks = []
        self.actions_alltasks = []

        self.proprio_data_by_task = dict()
        self.image_names_by_task = dict()

        self.datas = []
        for task_index, task_name in enumerate(self.task_names):
            task_dir = self.data_dir_path / task_name

            self.proprio_data_by_task[task_name] = dict()
            self.image_names_by_task[task_name] = dict()

            data_demo_idxs = np.arange(start_idx,start_idx+num_demos_per_task)

            for demo_idx, demo_dir in enumerate(task_dir.iterdir()):
                assert demo_dir.name.startswith('demo')
                if demo_idx in data_demo_idxs:
                    info_pickle = demo_dir / 'info.pickle'
                    with open(info_pickle, 'rb') as info_f:
                        demo_info = pickle.load(info_f)
                    demo_info['proprio'] = demo_info['observations'][:, :proprio_len]
                    del demo_info['observations']
                    self.proprio_data_by_task[task_name][demo_dir.name] = demo_info
                    self.image_names_by_task[task_name][demo_dir] = dict()

                    for camera_name in camera_names:
                        if not self.is_multiview_dataset:
                            demo_images = [f for f in demo_dir.iterdir() if f.suffix == '.png']
                        else:
                            img_dir = demo_dir / camera_name
                            demo_images = [f for f in img_dir.iterdir() if f.suffix == '.png']
                        # images have names 'img_t{}.png'
                        demo_images = sorted(demo_images, key=lambda x: int(x.name.split('img_t')[1].split('.')[0]))
                        self.image_names_by_task[task_name][demo_dir][camera_name] = demo_images
                        

                    use_multitemporal_sensors = self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use
                    if use_multitemporal_sensors:
                        static_cam_freq = self.multi_temporal_sensors.i3_freq
                        hand_cam_freq = self.multi_temporal_sensors.ih_freq
                    else:
                        static_cam_freq = 1
                        hand_cam_freq = 1
                    
                    for step in range(demo_info['proprio'].shape[0]):
                        demo_image_path = dict()
                        for camera_name in camera_names:
                            cam_freq = static_cam_freq if self.is_static_camera(camera_name) else hand_cam_freq
                            last_sensor_step = (int(step // cam_freq))
                            demo_image_path[camera_name + '_last_step'] = (
                                self.image_names_by_task[task_name][demo_dir][camera_name][last_sensor_step])

                        if self.append_object_mask == 'mdetr':
                            demo_image_path['object_mask'] = demo_dir / 'heatmap' / 'img_t0_mdetr.png'
                        elif self.append_object_mask == 'owl_vit':
                            demo_image_path['object_mask'] = demo_dir / 'heatmap' / 'img_t0_owlvit.png'

                        di = DataItem(task_name, task_index, demo_dir.name, demo_image_path, step, None)
                        self.datas.append(di)

                # if demo_idx >= num_demos_per_task:
                #     break

        self._has_png_images = True

    def is_static_camera(self, cam_name):
        if 'hand' in cam_name:
            return False
        assert cam_name == 'left_cap2', "Only one camera for now"
        return True

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data: DataItem = self.datas[idx]
        task_name = data.task
        task_idx = data.task_index
        demo_name = data.demo_dir
        t = data.step

        task_onehot = np.zeros((self.num_tasks))
        task_onehot[task_idx] = 1
        task_name = self.task_names[task_idx]

        info = self.proprio_data_by_task[task_name][demo_name]
        proprio = info['proprio'][t]
        
        if self.action_type == 'delta_obs_pos':
            action = info['actions_obs'][t]
        elif self.action_type == 'des_tsc':
            action = info['actions_des'][t]
        elif self.action_type == 'delta_des_tsc':
            action = info['actions_des_delta'][t]
        else:
            raise NotImplementedError("Action type not available")

        if self.normalize_action:
            action = np.clip(action, self.min_action, self.max_action)
            action = -1 + 2*(action - self.min_action) / (self.max_action - self.min_action)

        sample = {
            'task': task_name,
            'task_enc': torch.FloatTensor(task_onehot),
            'proprio': torch.FloatTensor(proprio),
            'expert_actions': torch.FloatTensor(action),
        }

        for camera_name in self.camera_names:
            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors.use:
                # NOTE: For hand cameras last step will be first step
                # TODO(saumya): remove hardcoded [:,:,:3]
                image = np.asarray(Image.open(data.img_path[camera_name + '_last_step']))[:,:,:3]
            else:
                # TODO(saumya): remove hardcoded [:,:,:3]
                image = np.asarray(Image.open(data.img_path[camera_name]))[:,:,:3]

            if self.append_object_mask is not None and camera_name == 'left_cap2':
                object_mask = np.asarray(Image.open(data.img_path['object_mask']))
                image = np.append(image, object_mask[:,:,None], axis=2)
                # sample.update({'object_mask': torch.FloatTensor(object_mask)})

            sample.update({camera_name: torch.FloatTensor(image)})

        if self.transform:
            sample = self.transform(sample)
        return sample
    