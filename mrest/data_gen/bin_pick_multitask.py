import numpy as np
import hydra
import pickle
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig
from pathlib import Path

from collections import OrderedDict
from itertools import product
from typing import Any, List, Mapping, Optional

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_bin_picking_v2_multitask import SawyerBinPickingMultiTaskEnvV2
from metaworld.policies.sawyer_bin_picking_multitask_v2_policy import SawyerBinPickingMultiTaskV2Policy

from mrest.data_gen.create_metaworld_scripted_data import trajectory_summary
from PIL import Image


def convert_object_color_to_str(color: str) -> str:
    normal_colors = ['green', 'orange', 'yellow', 'pink', 'purple', 'black', 'grey']
    if color in normal_colors:
        return f'{color} block'
    
    composite_colors = {
        'dark_chocolate': 'chocolate',
        'golden_yellow': 'golden',
    }
    return f'{composite_colors[color]} block'


def convert_object_size_to_str(obj_name: str) -> str:
    # xyz
    objects = {
        'small_cube': 'small block',
        'medium_cube': 'medium block',
        'large_cube': 'large block',
        'small_medium_cuboid': 'small and medium block',
        'large_small_cuboid': 'large and small block',
        'small_large_cuboid': 'small and large block',
        'large_small_cuboid': 'large and small block',
    }
    return objects[obj_name]

def select_other_obj_configs(color: str, obj_size: str, 
                             all_colors: List[str], 
                             all_sizes: List[str]):
    other_colors = [c for c in all_colors if c != color]
    other_sizes = [s for s in all_sizes if s != obj_size]
    other_obj_configs = []
    for other_obj_colors in product(other_colors, other_colors):
        for other_other_obj_sizes in product(other_sizes, other_sizes):
            other_obj_config = [
                { 'color': other_obj_colors[0], 'size': other_other_obj_sizes[0] },
                { 'color': other_obj_colors[1], 'size': other_other_obj_sizes[1] },
            ]
            other_obj_configs.append(other_obj_config)

    return other_obj_configs


def generate_env_configs(start_bin_locations: List[int],
                         target_colors: List[str],
                         target_sizes: List[str],
                         color_task_template: str,
                         size_task_template: str,
                         all_colors: List[str],
                         all_sizes: List[str],
                         max_other_obj_configs_to_select: Optional[int] = None):

    env_configs = []

    for start_bin_index in start_bin_locations:
        goal_bin_locations = [2, 3] if start_bin_index in [0, 1] else [0, 1]
        for goal_bin_index in goal_bin_locations:

            for color in target_colors:
                for obj_size in target_sizes:

                    task_command_color = color_task_template.format(
                        obj_color=convert_object_color_to_str(color))
                    task_command_size = size_task_template.format(
                        obj_size=convert_object_size_to_str(obj_size))

                    other_obj_configs = select_other_obj_configs(
                        color, obj_size, all_colors, all_sizes)
                    
                    np.random.shuffle(other_obj_configs)
                    if max_other_obj_configs_to_select:
                        other_obj_configs = other_obj_configs[:max_other_obj_configs_to_select]
                    
                    for other_obj_config in other_obj_configs:
                        env_config = OrderedDict(
                            start_bin_index=start_bin_index,
                            goal_bin_index=goal_bin_index,
                            objA_config=OrderedDict(
                                size=obj_size,
                                color=color
                            ),
                            objB_config=OrderedDict(
                                size=other_obj_config[0]['size'],
                                color=other_obj_config[0]['color'],
                            ),
                            objC_config=OrderedDict(
                                size=other_obj_config[1]['size'],
                                color=other_obj_config[1]['color'],
                            ),
                            task_command_color=task_command_color,
                            task_command_size=task_command_size,
                        )
                        env_configs.append(env_config)

    return env_configs

def collect_demo_data_for_env_config(env_config: Mapping[str, Any], num_demos: int): 
    env = SawyerBinPickingMultiTaskEnvV2(
        0,
        env_config['start_bin_index'],
        env_config['goal_bin_index'],
        env_config['objA_config'],
        env_config['objB_config'],
        env_config['objC_config'],
        )
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    policy = SawyerBinPickingMultiTaskV2Policy()

    action_noisy_percent = 0.0

    all_demo_dicts = []
    successes = 0

    for demo_idx in range(num_demos):
        traj_success = 0
        while not traj_success:
            # Wait for 5 steps after first completion
            traj_info = trajectory_summary(
                env, policy, action_noisy_percent, render=True,
                end_on_success=True, demo_idx=demo_idx, 
                end_on_success_followup_steps=5,
                use_opencv_to_render=True)
            traj_success = int(traj_info[0])
        successes += float(traj_info[0])
        demo_dict = traj_info[4]
        assert isinstance(demo_dict, dict), f"Not a valid demo dict: {type(demo_dict)}"

        all_demo_dicts.append(demo_dict)

    print(f"Total success: {successes:.4f}")
    return all_demo_dicts

def save_demos(demo_dicts: List[Mapping[str, Any]], save_dir: Path, env_name: str, camera_names=['left_cap2']):
    '''Save list of demos (as dictionary).'''
        # with open(save_path, 'wb') as pkl_f:
        #     pickle.dump(demo_dicts, pkl_f, protocol=4)

    if len(camera_names) == 1:
        save_dir = save_dir / 'left_cap2' / env_name
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir()

        for i in range(len(demo_dicts)):
            save_dir_demo = save_dir / f'demo{i}'
            save_dir_demo.mkdir()

            non_image_data = {
                'observations': demo_dicts[i]['observations'],
                'rewards': demo_dicts[i]['rewards'],
                'actions': demo_dicts[i]['actions'],
            }
            if 'actions_with_noise' in demo_dicts[i]:
                non_image_data['actions_with_noise'] = demo_dicts[i]['actions_with_noise']

            with open(save_dir_demo / 'info.pickle', 'wb') as f:
                pickle.dump(non_image_data, f)
            
            T = demo_dicts[i]['observations'].shape[0]
            for t in range(T):
                tmp_img = Image.fromarray(demo_dicts[i][camera_names[0]][t].astype(np.uint8))
                tmp_img.save(save_dir_demo / f'img_t{t}.png')
    else:
        save_dir = save_dir / env_name
        save_dir.parent.mkdir(exist_ok=True)
        save_dir.mkdir()

        for i in range(len(demo_dicts)):
            save_dir_demo = save_dir / f'demo{i}'
            save_dir_demo.mkdir()

            non_image_data = {
                'observations': demo_dicts[i]['observations'],
                'rewards': demo_dicts[i]['rewards'],
                'actions': demo_dicts[i]['actions'],
            }

            with open(save_dir_demo / 'info.pickle', 'wb') as f:
                pickle.dump(non_image_data, f)
            
            T = demo_dicts[i]['observations'].shape[0]
            for camera_name in camera_names:
                save_dir_imgs = save_dir_demo / camera_name
                save_dir_imgs.mkdir()
                for t in range(T):
                    tmp_img = Image.fromarray(demo_dicts[i][camera_name][t].astype(np.uint8))
                    tmp_img.save(save_dir_imgs / f'img_t{t}.png')

    print(f"Did save demos at: {save_dir}")
    

def recursively_filter_noyaml_subdicts(data):
    '''Certain key-values cannot be stored in yaml files. Hence recursively filter them out.
    
    These are marked with the _no_yaml flag.
    '''
    if not isinstance(data, dict) and not isinstance(data, OrderedDict):
        return data

    no_yaml = data.get('_no_yaml', False)
    if no_yaml:
        return None

    no_yaml_data = {}
    for k, v in data.items():
        no_yaml_data[k] = recursively_filter_noyaml_subdicts(v)
        if no_yaml_data[k] is None:
            del no_yaml_data[k]
    return no_yaml_data


def save_env_config(env_config, data_dir: Path, config_name: str):
    '''Save env config as pickle and yaml file for quick read.'''
    pkl_path = data_dir / (config_name + '.pkl')
    with open(pkl_path, 'wb') as pkl_f:
        pickle.dump(env_config, pkl_f, protocol=4)
    yaml_path = data_dir / (config_name + '.yaml')
    with open(yaml_path, 'w') as yaml_f:
        # yaml.dump(env_config, yaml_f, default_flow_style=False)
        env_config = recursively_filter_noyaml_subdicts(env_config)
        yaml.safe_dump(env_config, yaml_f, default_flow_style=False)


@hydra.main(version_base="1.1.0", config_name="config",config_path="config")
def main(config: DictConfig) -> None:
    task_template_color = 'pick up the {obj_color:s} and place it in goal.'
    task_template_size = 'pick up the {obj_size:s} and place it in goal.'

    all_colors = config.all_colors
    all_sizes = config.all_sizes

    train_env_configs = generate_env_configs(
        config.train_start_bin_locations,
        config.train_colors,
        config.train_sizes,
        task_template_color,
        task_template_size,
        all_colors,
        all_sizes,
        max_other_obj_configs_to_select=8,
    )
    # breakpoint()
    print(f'num of train env configs: {len(train_env_configs)}')
    main_data_dir = Path(config.data.data_dir)
    if not main_data_dir.exists():
        main_data_dir.mkdir()
    
    # if config.data.generate:
    #     # Save env configs
    #     to_save_train_env_configs = OrderedDict()
    #     train_data_dir = main_data_dir / 'train'
    #     if not train_data_dir.exists():
    #         train_data_dir.mkdir()

    #     for train_env_idx, env_config in enumerate(train_env_configs):
    #         demos = collect_demo_data_for_env_config(env_config, 2)
    #         env_name = f'env_{train_env_idx:04d}'
    #         to_save_train_env_configs[env_name] = env_config
    #         save_demos(demos, train_data_dir, env_name)
    #     save_env_config(to_save_train_env_configs, main_data_dir, 'train_env_configs')

    eval_env_configs = generate_env_configs(
        config.eval_start_bin_locations,
        config.eval_colors,
        config.eval_sizes,
        task_template_color,
        task_template_size,
        all_colors,
        all_sizes,
        max_other_obj_configs_to_select=4,
    )
    print(f'num of eval env configs: {len(eval_env_configs)}')
    if config.data.generate:
        # Save env configs
        to_save_eval_env_configs = OrderedDict()
        eval_data_dir = main_data_dir / 'eval'
        if not eval_data_dir.exists():
            eval_data_dir.mkdir()

        for eval_env_idx, env_config in enumerate(eval_env_configs):
            demos = collect_demo_data_for_env_config(env_config, 1)
            env_name = f'eval_env_{eval_env_idx:04d}'
            to_save_eval_env_configs[env_name] = env_config
            save_demos(demos, eval_data_dir, env_name)
        save_env_config(to_save_eval_env_configs, main_data_dir, 'eval_env_configs')


if __name__ == "__main__":
    main()
