import pickle
import yaml, os
import numpy as np
import hydra

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Mapping
from PIL import Image
from tqdm import trange

from mrest.data_gen.create_metaworld_scripted_data import trajectory_summary

from metaworld.envs.mujoco.franka_pickandplace.franka_pick_place_multitask_env import FrankaPickPlaceMultitaskEnv
from metaworld.envs.mujoco.franka_pickandplace.franka_pick_place_multitask_env_fix_orient import FrankaPickPlaceMultitaskFixOrientEnv
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_mocap import SawyerPickAndPlaceMultiTaskEnvV2

from mrest.data_gen.abr_policies.pick_place_mj_abr_policy import PickPlaceMjAbrPolicy
from mrest.data_gen.abr_policies.pick_place_mj_abr_policy_xyz import PickPlaceMjAbrPolicyXYZ
from mrest.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy

from bin_pick_multitask import save_env_config, save_demos

def collect_demo_data_for_env_config(env_config: Mapping[str, Any], num_demos: int): 
    env = FrankaPickPlaceMultitaskFixOrientEnv(env_config)
    env._set_task_called = True

    action_noisy_percent = 0.0

    all_demo_dicts = []
    successes = 0

    for demo_idx in trange(num_demos):
        traj_success = 0
        while not traj_success:
            # policy = PickPlaceMjAbrMocapPolicy(env.robot_config, skill=env_config['skill'], target_object=env_config['target_object'])
            policy = PickPlaceMjAbrPolicyXYZ(env.robot_config, skill=env_config['skill'], target_object=env_config['target_object'])
            # Wait for 5 steps after first completion
            traj_info = trajectory_summary(
                env, policy, action_noisy_percent, render=True,
                end_on_success=True, demo_idx=demo_idx, 
                end_on_success_followup_steps=5,
                use_opencv_to_render=False)
            traj_success = int(traj_info[0])
        successes += float(traj_info[0])
        demo_dict = traj_info[4]
        assert isinstance(demo_dict, dict), f"Not a valid demo dict: {type(demo_dict)}"

        all_demo_dicts.append(demo_dict)

    print(f"Total success: {successes:.4f}")
    return all_demo_dicts

def generate_env_configs(config, skill):
    """
    For a certain skill, we sample over target objects for each env.
    With respect to env physical properties, we sample only locations of objects for now.
    TODO: Change object colors/tectures
    TODO: Change object sizes
    """
    env_configs = []

    for target_object in config.target_objects:
        for block_size in config.all_block_sizes:
            for block_color in config.all_block_colors:
                for coke_color in config.all_coke_colors:
                    for pepsi_color in config.all_pepsi_colors:
                        for milk_color in config.all_milk_colors:
                            for bread_color in config.all_bread_colors:
                                for bottle_color in config.all_bottle_colors:
                                    block_config=OrderedDict(
                                        size=block_size,
                                        color=block_color
                                    )
                                    coke_config=OrderedDict(
                                        size='medium',
                                        color=coke_color
                                    )
                                    pepsi_config=OrderedDict(
                                        size='medium',
                                        color=pepsi_color
                                    )
                                    milk_config=OrderedDict(
                                        size='medium',
                                        color=milk_color
                                    )
                                    bread_config=OrderedDict(
                                        size='medium',
                                        color=bread_color
                                    )
                                    bottle_config=OrderedDict(
                                        size='medium',
                                        color=bottle_color
                                    )

                                    if 'standard' in config.language:
                                        task_command_lang=f'{skill} {target_object}'
                                    else:
                                        synonym = config['synonyms'][target_object]
                                        task_command_lang=f'{skill} {synonym}'

                                    target_object_config = eval(f'{target_object}_config')
                                    target_object_color = target_object_config['color']
                                    target_object_size = target_object_config['size']
                                    env_config = OrderedDict(
                                            target_object=target_object,
                                            block_config=block_config,
                                            coke_config=coke_config,
                                            pepsi_config=pepsi_config,
                                            milk_config=milk_config,
                                            bread_config=bread_config,
                                            bottle_config=bottle_config,
                                            
                                            skill=skill,
                                        task_command_color=f'{skill} {target_object_color} object',
                                        task_command_size=f'{skill} {target_object_size} object',
                                        task_command_type=f'{skill} {target_object}',
                                        task_command_lang=task_command_lang,
                                        )
                                    env_configs.append(env_config)
    return env_configs

def get_env_name(env_config, data_config):
    if 'target' in data_config.vary:
        target = env_config['target_object']
        return f'env_{data_config.vary}_{target}'
    elif 'lang' in data_config.vary:
        target = env_config['target_object']
        return f'env_{data_config.vary}_{target}'
    elif 'color' in data_config.vary:
        color = env_config['block_config']['color']
        return f'env_{data_config.vary}_{color}'
    else:
        raise NotImplementedError("Dataset variat not implemented.")

def collect_data(config, data_config, skill, main_data_dir, data_type='train', num_demos=1):
    env_configs = generate_env_configs(config, skill)
    print(f'NUM OF {data_type} ENV CONFIGS: {len(env_configs)}')
    
    if data_config.save:
        # Save env configs
        to_save_env_configs = OrderedDict()
        data_dir = main_data_dir / data_type
        if not data_dir.exists():
            os.makedirs(data_dir)

        for env_idx, env_config in enumerate(env_configs):
            env_name = get_env_name(env_config, data_config)
            print(F"Collecting data for {data_type} env {env_idx}/{len(env_configs)}: {env_name}")
            demos = collect_demo_data_for_env_config(env_config, num_demos)
            to_save_env_configs[env_name] = env_config
            save_demos(demos, data_dir, env_name)
        save_env_config(to_save_env_configs, main_data_dir, f'{data_type}_env_configs')


@hydra.main(version_base="1.1.0", config_name="franka_pickplace_data_gen_cfg_vary_target",config_path="config")
def main(config: DictConfig) -> None:

    skill_name_short = config.skills.use.lower().replace(" ", "")
    dataset_name = f'{config.data.prefix}_{skill_name_short}_{config.data.vary}'
    print(f"Dataset name: {dataset_name}")

    main_data_dir = Path(config.data.data_dir) / dataset_name
    if not main_data_dir.exists():
        os.makedirs(main_data_dir)
    
    # Collect and save train data
    print("================Collecting data for train environments================")
    collect_data(config.train, config.data, config.skills.use, main_data_dir, data_type='train', num_demos=config.data.num_demos_per_train_env)

    # Collect and save train data
    print("================Collecting data for eval environments================")
    collect_data(config.eval, config.data, config.skills.use, main_data_dir, data_type='eval', num_demos=config.data.num_demos_per_eval_env)
    
    # Saving the config used for data generation
    with open(main_data_dir/'data_cfg.yaml', "w") as f:
        OmegaConf.save(config, f)

if __name__ == "__main__":
    main()