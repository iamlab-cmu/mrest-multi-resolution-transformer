import copy
import pickle
import yaml, os
import itertools
import numpy as np
import hydra
import random
from dataclasses import dataclass

from omegaconf import OmegaConf, DictConfig, ListConfig
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union
from PIL import Image
from tqdm import trange

from mrest.data_gen.create_metaworld_scripted_data import trajectory_summary

from metaworld.envs.mujoco.franka_pickandplace.franka_pick_place_multitask_env import FrankaPickPlaceMultitaskEnv
from metaworld.envs.mujoco.franka_pickandplace.franka_pick_place_multitask_env_fix_orient import FrankaPickPlaceMultitaskFixOrientEnv
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_mocap import SawyerPickAndPlaceMultiTaskEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_gen_mocap import SawyerPickAndPlaceMultiTaskGenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_multiobj_multitask_env_gen_mocap import SawyerMultiObjectMultiTaskGenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_multiobj_multitask_multishape_env_gen import SawyerMultiObjectMultiTaskMultiShapeGenEnvV2
from metaworld.envs.mujoco2_3.sawyer_xyz.sawyer_pick_place_multitask_procedural import SawyerPickAndPlaceMultiTaskEnvProcV2

from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy

from mrest.data_gen.abr_policies.pick_place_mj_abr_policy import PickPlaceMjAbrPolicy
from mrest.data_gen.abr_policies.pick_place_mj_abr_policy_xyz import PickPlaceMjAbrPolicyXYZ
from mrest.data_gen.abr_policies.pick_place_mj_abr_policy_mocap import PickPlaceMjAbrMocapPolicy

from mrest.data_gen.bin_pick_multitask import save_env_config, save_demos
from metaworld.policies import SawyerMultitaskV2Policy


def recursively_get_dict_from_omegaconf(conf):

    if isinstance(conf, ListConfig):
        return [vi for vi in conf]
    
    if not isinstance(conf, DictConfig):
        return conf

    conf_dict = OrderedDict()
    for k, v in conf.items():
        conf_dict[k] = recursively_get_dict_from_omegaconf(v)
    
    return conf_dict


def collect_demo_data_for_env_config(env_config: Mapping[str, Any], num_demos: int): 
    # Manually set env for skill generalization. Should probably automate this, but it's fine for now.
    if 'multiobj' in env_config:
        if 'procedural' in env_config:
            env = SawyerPickAndPlaceMultiTaskEnvProcV2(env_config, data_collection=True,)
        else:
            if env_config.get('has_small_objects', False):
                env = SawyerMultiObjectMultiTaskMultiShapeGenEnvV2(env_config, data_collection=True,)
            else:
                env = SawyerMultiObjectMultiTaskGenEnvV2(env_config, data_collection=True,)
    else:
        env = SawyerPickAndPlaceMultiTaskGenEnvV2(env_config, data_collection=True,)

    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    action_noisy_percent = 0.07 if np.random.uniform() > 0.2 else 0.0

    all_demo_dicts = []
    successes = 0

    for demo_idx in trange(num_demos):
        traj_success = 0
        while not traj_success:
            # policy = PickPlaceMjAbrMocapPolicy(env.robot_config, skill=env_config['skill'], target_object=env_config['target_object'])
            if 'multiobj' in env_config:
                policy = SawyerMultitaskV2Policy(env_config['skill'], env_config['target_object'])
            else:
                if env_config.get('task_buttonpress'):
                    policy = SawyerButtonPressV2Policy()
                else:
                    policy = PickPlaceMjAbrMocapPolicy(skill=env_config['skill'],
                                                       target_object=env_config['target_object'])
            # Wait for 5 steps after first completion
            traj_info = trajectory_summary(
                env, policy, action_noisy_percent, render=True,
                end_on_success=True, demo_idx=demo_idx,
                end_on_success_followup_steps=5,
                use_opencv_to_render=False,
                camera_names=env_config.get('camera_names', ['left_cap2']))
            traj_success = int(traj_info[0])
        successes += float(traj_info[0])
        demo_dict = traj_info[4]
        assert isinstance(demo_dict, dict), f"Not a valid demo dict: {type(demo_dict)}"

        all_demo_dicts.append(demo_dict)

    print(f"Total success: {successes:.4f}")
    return all_demo_dicts


@dataclass
class MultiObjMultiSkillConfigInstance:
    target_obj: str
    distractor_big_objects: List[str]
    distractor_medium_objects: List[str]
    distractor_blocks: List[str]
    stack_on_obj: Optional[str] = None
    distractor_small_objects: List[str] = None
    distractor_sticks: List[str] = None


def generate_multiobj_multiskill_configs(
    target_objs: List[str],
    distractor_config: Mapping[str, Any],
    all_objects: List[str],
    stack_on_objs: Optional[List[str]] = None,
    small_objects: Optional[List[str]] = None,
    auxiliary_objects: Optional[List[str]] = None,) -> List[MultiObjMultiSkillConfigInstance]:
    """Generate object variations.
    
    Args:

    auxiliary_objects: List of auxiliary objects that are required to be in the current scene.
    """

    final_block_configs = []

    def _split_objects_into_types(target_obj: str, objects: List[str],
                                  auxiliary_objs: Optional[List[str]] = None,):
        big_objects, medium_objects, small_objects, curr_blocks = [], [], [], []
        curr_sticks = []

        auxiliary_objs = auxiliary_objs or []

        for obj in itertools.chain([target_obj], objects, auxiliary_objs):
            if obj in ['door', 'door_small', 'window', 'drawer', 'drawer_small']:
                big_objects.append(obj)
            elif obj in ['peg', 'RoundNut', 'faucetBase', 'red_mug', 'white_mug', 'blue_mug', 'reebok_black_shoe', 'reebok_blue_shoe', 'green_shoe', 'pink_heel']:
                medium_objects.append(obj)
            elif obj in ['milk', 'coke', 'pepsi', 'bread']:
                small_objects.append(obj)
            elif obj.startswith('stick'):
                curr_sticks.append(obj) 
            elif obj in all_objects:
                curr_blocks.append(obj)
            else:
                raise ValueError(f'Invalid object: {obj}')

        return {
            'big_objects': big_objects,
            'medium_objects': medium_objects,
            'small_objects': small_objects,
            'blocks': curr_blocks,
            'sticks': curr_sticks,
        }
    
    distractor_objs = distractor_config['distractors'].copy()
    for obj in target_objs:
        # remove target object from distractor list
        distractor_config['distractors'] = distractor_objs.copy()
        if obj in distractor_config['distractors']:
            distractor_config['distractors'].remove(obj)
            
        # Find any distractor blocks for this config.
        if 'blocks' in distractor_config:
            possible_distractor_blocks = set(distractor_config.blocks) - {obj}

        if 'sticks' in distractor_config:
            possible_distractor_sticks = set(distractor_config.sticks) - {obj}

        possible_small_objects = set(small_objects) - {obj}
        
        # Remove any auxliary objects from distractors
        for aux_obj in (auxiliary_objects or []):
            if 'blocks' in distractor_config:
                possible_distractor_blocks = possible_distractor_blocks - {aux_obj}
            if 'sticks' in distractor_config:
                possible_distractor_sticks = possible_distractor_sticks - {aux_obj}
            possible_small_objects = possible_small_objects - {aux_obj}
            assert aux_obj not in distractor_config['distractors'], (
                f'Object specified both as auxiliary and distractor: {aux_obj}')


        # Is this a stacking skill? If yes, then find which blocks can we stack on.
        if stack_on_objs:
            possible_stack_on_objs = [o for o in stack_on_objs if o != obj]
            np.random.shuffle(possible_stack_on_objs)

        # If we are performing stack skill then select num_distractor_blocks + 1.
        # We will choose the first block as the stack_on_block
        num_distractor_blocks = 0 if stack_on_objs is None else 1
        non_block_distractors = []
        num_distractors_stick = 0
        num_distractors_small_obj = 0
        for distractor in distractor_config['distractors']:
            if distractor == 'block':
                num_distractor_blocks += 1
            elif distractor == 'stick':
                num_distractors_stick += 1
            elif distractor == 'small_obj':
                num_distractors_small_obj += 1
            else:
                assert distractor != obj and distractor in all_objects, 'Distractor not in objects'
                non_block_distractors.append(distractor)

        if num_distractor_blocks:
            distractor_blocks = list(itertools.combinations(possible_distractor_blocks, num_distractor_blocks))
            np.random.shuffle(distractor_blocks)

            if stack_on_objs:
                distractor_blocks = [block_set for block_set in distractor_blocks if block_set[0] in stack_on_objs]

        if num_distractors_stick:
            distractor_sticks = list(itertools.combinations(possible_distractor_sticks, num_distractors_stick))
            np.random.shuffle(distractor_sticks)
        else:
            distractor_sticks = []
        
        if num_distractors_small_obj:
            distractor_small_objs = list(itertools.combinations(possible_small_objects, num_distractors_small_obj))
            np.random.shuffle(distractor_small_objs)
        else:
            distractor_small_objs = []

        objects_by_type = _split_objects_into_types(obj, non_block_distractors, auxiliary_objs=auxiliary_objects)
        big_objects = objects_by_type['big_objects']
        medium_objects = objects_by_type['medium_objects']

        required_small_objects = objects_by_type['small_objects']
        required_blocks = objects_by_type['blocks']
        required_sticks = objects_by_type['sticks']

        if 'num_distractor_block_variations' in distractor_config:
            num_distractor_variations = (distractor_config.num_distractor_block_variations
                                         if num_distractor_blocks > 0 else 0)
        elif 'num_distractor_variations' in distractor_config:
            num_distractors = num_distractor_blocks + num_distractors_stick + num_distractors_small_obj
            num_distractor_variations = (distractor_config.num_distractor_variations
                                         if num_distractors > 0 else 0)
        else:
            raise ValueError('Config should specify how many distractor variations we want.')

        if num_distractor_variations:
            for i in range(num_distractor_variations):
                config_stack_on_obj = distractor_blocks[i][0] if stack_on_objs else None
                # All distractor blocks in this env.
                all_blocks = list(distractor_blocks[i]) if num_distractor_blocks and len(distractor_blocks) > i else []
                all_sticks = list(distractor_sticks[i]) if len(distractor_sticks) > i else []
                all_small_objs = list(distractor_small_objs[i]) if len(distractor_small_objs) > i else []

                # If we have some required blocks, sticks, small objects then add them to the distractors to get
                # the entire list of blocks, sticks, small_objs.

                if len(required_blocks):
                    all_blocks = required_blocks + all_blocks
                if len(required_sticks):
                    all_sticks = required_sticks + all_sticks
                if len(required_small_objects):
                    all_small_objs = required_small_objects + all_small_objs
                
                config = MultiObjMultiSkillConfigInstance(
                        obj, big_objects, medium_objects, all_blocks,
                        stack_on_obj=config_stack_on_obj,
                        distractor_small_objects=all_small_objs,
                        distractor_sticks=all_sticks,)
                final_block_configs.append(config)

        else:
            assert stack_on_objs is None, 'Stack skill cannot have no stack_on_obj'

            # No distractor block
            config = MultiObjMultiSkillConfigInstance(obj, big_objects, medium_objects, [])
            final_block_configs.append(config)

    return final_block_configs


@dataclass
class BlockConfigInstance:
    """Class for keeping track of an item in inventory."""
    target_obj: str
    all_blocks: List[str]
    stack_on_obj: Optional[str] = None


def generate_single_target_obj_configs(
    target_objs: List[str],
    distractor_blocks: List[str],
    num_distractor_blocks: int = 3,
    num_distractor_block_variations: int = 1,
    is_target_object_a_block: bool = False) -> List[BlockConfigInstance]:
    """Generate object variations."""

    final_block_configs = []

    for obj in target_objs:
        possible_distractor_blocks = set(distractor_blocks) - {obj}
        other_blocks = list(itertools.combinations(possible_distractor_blocks, num_distractor_blocks))
        np.random.shuffle(other_blocks)

        for i in range(num_distractor_block_variations):
            all_blocks = [obj] + [*other_blocks[i]] if is_target_object_a_block else other_blocks[i]
            final_block_configs.append(BlockConfigInstance(obj, all_blocks))

    return final_block_configs


def generate_stack_skill_obj_configs(
    target_objs, stack_on_objs, distractor_blocks,
    num_distractor_blocks: int = 2,
    num_stack_on_objects_for_one_target: int = 2,
    num_distractor_block_variations: int = 2,) -> List[BlockConfigInstance]:

    final_block_configs = []

    for obj in target_objs:

        possible_stack_on_objs = [o for o in stack_on_objs if o != obj]
        np.random.shuffle(possible_stack_on_objs)

        for stack_on_obj in possible_stack_on_objs[:num_stack_on_objects_for_one_target]:
            possible_distractor_blocks = set(distractor_blocks) - set([obj, stack_on_obj])
            other_blocks = list(itertools.combinations(possible_distractor_blocks, num_distractor_blocks))

            np.random.shuffle(other_blocks)
            for i in range(num_distractor_block_variations):
                all_blocks = [obj, stack_on_obj] + [*other_blocks[i]]
                final_block_configs.append(
                    BlockConfigInstance(obj, all_blocks, stack_on_obj=stack_on_obj))

    return final_block_configs


def generate_multiobj_multiskill_env_configs(config: DictConfig, skill_cfg: DictConfig, procedural=False):
    skills = skill_cfg.all_skills if skill_cfg.use == 'all' else [skill_cfg.use]

    env_configs = []

    all_objects = config['all_objects']['objects'] + config['all_objects']['blocks']
    # small_objects refers to coke, pepsi etc. (These were not present in the first round of data collection.)
    small_objects = config['all_objects'].get('small_objects', [])
    has_small_objects = config.get('has_small_objects', False)

    sample_single_description = config.get('sample_single_description', True)

    for skill in skills:
        # Auxiliary objects are objects required by the skill that are independent of target objects.
        # This is only used for composite skills
        auxiliary_objects = config[skill].get('auxiliary_objects', [])
        # Create empty auxiliary objects
        
        if len(auxiliary_objects) == 0:
            auxiliary_objects = [[] for _ in range(len(config[skill]))]
            auxobj_and_distractor_iter = zip(auxiliary_objects, 
                                      [k for k in config[skill].distractor_configs.keys()])
        else:
            auxobj_and_distractor_iter = itertools.product(
                auxiliary_objects, 
                [k for k in config[skill].distractor_configs.keys()])

        # for distractor_cfg_key, distractor_cfg in config[skill].distractor_configs.items():
        for auxiliary_objs, distractor_cfg_key in auxobj_and_distractor_iter:

            distractor_cfg = config[skill].distractor_configs[distractor_cfg_key]

            if 'stack' in skill:
                stack_on_objs = config[skill]['stack_on_objects']
            else:
                stack_on_objs = None

            target_objs = config[skill]['target_blocks'] + config[skill]['target_objects']
            if 'target_sticks' in config[skill]:
                target_objs += config[skill]['target_sticks']
            # Generate configs for skills that involve single target object
            obj_configs = generate_multiobj_multiskill_configs(
                target_objs, distractor_cfg, all_objects, stack_on_objs=stack_on_objs,
                small_objects=small_objects,
                auxiliary_objects=auxiliary_objs)
            
            for env_obj_config in obj_configs:

                skill_desc = sample_description(config[skill]["skill_desc"])
                target_object = env_obj_config.target_obj
                # TODO(Mohit): Do not sample here choose and save all of them as a list.
                if sample_single_description:
                    target_object_desc = sample_description(config['object_description'][target_object])
                    task_command_type = get_task_command(skill_desc, target_object_desc)
                else:
                    task_command_type = []
                    for target_object_desc in config['object_description'][target_object]:
                        task_command_type.append(get_task_command(skill_desc, target_object_desc))

                if target_object.startswith('block_') or target_object.startswith('stick_'):
                    # NOTE: This is wrongly called distractor blocks. It is really all_blocks.
                    if len(env_obj_config.distractor_blocks) > 0:
                        target_obj_is_first_block = target_object == env_obj_config.distractor_blocks[0]
                    if len(env_obj_config.distractor_sticks) > 0:
                        target_obj_is_first_stick = target_object == env_obj_config.distractor_sticks[0]
                    assert target_obj_is_first_block or target_obj_is_first_stick
                    target_object_color = target_object.split('_')[1]
                    if 'stick' in target_object:
                        task_command_color = get_task_command(skill_desc, f'{target_object_color} stick')
                    else:
                        task_command_color = get_task_command(skill_desc, f'{target_object_color} object')
                else:
                    task_command_color = ''

                # Rename distractor blocks from (block_red, block_pink) to blockA, blockB
                block_configs = {}
                if (len(env_obj_config.distractor_blocks) and
                    env_obj_config.distractor_blocks[0].startswith('block_')):
                    block_suffixes = ['A', 'B', 'C', 'D']
                    distractor_blocks = []
                    for b_idx, b in enumerate(env_obj_config.distractor_blocks):
                        distractor_blocks.append(f'block{block_suffixes[b_idx]}')
                        block_configs[f'block{block_suffixes[b_idx]}_config'] = {'color': b}
                else:
                    distractor_blocks = env_obj_config.distractor_blocks

                # Rename distractor sticks from (block_red, block_pink) to blockA, blockB
                stick_configs = {}
                if (len(env_obj_config.distractor_sticks) and
                    env_obj_config.distractor_sticks[0].startswith('stick_')):
                    stick_suffixes = ['A', 'B']
                    distractor_sticks = []
                    for b_idx, b in enumerate(env_obj_config.distractor_sticks):
                        distractor_sticks.append(f'stick{stick_suffixes[b_idx]}')
                        stick_configs[f'stick{stick_suffixes[b_idx]}_config'] = {'color': b}
                else:
                    distractor_sticks = env_obj_config.distractor_sticks
                
                target_object_name = target_object
                if target_object_name.startswith('block_'):
                    target_object_name = 'blockA'
                elif target_object_name.startswith('stick_'):
                    target_object_name = 'stickA'

                if len(auxiliary_objs) > 0:
                    auxiliary_obj_name = auxiliary_objs[0]
                    if auxiliary_obj_name.startswith('block_'):
                        auxiliary_obj_name = 'blockA'
                    elif auxiliary_obj_name.startswith('stick_'):
                        auxiliary_obj_name = 'stickA'
                else:
                    auxiliary_obj_name = ''

                env_config = OrderedDict(
                        # First block is always the target object in our case now
                        target_object=target_object_name,
                        big_objects=env_obj_config.distractor_big_objects,
                        medium_objects=env_obj_config.distractor_medium_objects,
                        blocks=distractor_blocks,

                        sticks=distractor_sticks,
                        small_objects=env_obj_config.distractor_small_objects,
                        has_small_objects=has_small_objects,

                        # Auxiliary_objects:
                        auxiliary_objects=auxiliary_obj_name,

                        # Used in env_names
                        distractor_cfg_key=distractor_cfg_key,

                        task_command_color=task_command_color,
                        # task_command_size='', # TODO

                        skill=skill,
                        # TODO: For now we set the skill as the task can change this later?
                        task=skill,
                        task_command_type=task_command_type,
                        # task_command_lang=task_command_lang,  # TODO

                        # Only use block objects?
                        num_demos_per_env=distractor_cfg.num_demos_per_env,
                        update_block_colors=True,
                        update_stick_colors=True,
                        multiobj=True,
                        procedural=procedural,
                        randomize_medium_object_colors=True,
                        camera_names=list(config.get('camera_names', ['left_cap2'])),
                    )
                if distractor_cfg.get('target_object_color_cfg'):
                    env_config['target_object_color_cfg'] = recursively_get_dict_from_omegaconf(
                        distractor_cfg['target_object_color_cfg'])

                env_config.update(block_configs)
                env_config.update(stick_configs)

                if ('stack' in skill) or ('pick_and_place' in skill and ('left' in skill or 'right' in skill or 'back' in skill or 'front' in skill)):
                    # loop through which object to stack on
                    assert env_obj_config.stack_on_obj is not None and target_object != env_obj_config.stack_on_obj

                    stack_on_object = env_obj_config.stack_on_obj
                    assert stack_on_object in env_obj_config.distractor_blocks, (
                            'Stack on object not found in env blocks')
                    assert env_obj_config.distractor_blocks.index(stack_on_object) == 1, (
                            'stack on object should have index 1')
                    env_config['stack_on_object'] = 'blockB'

                    if sample_single_description:
                        stack_on_object_desc = sample_description(config['object_description'][stack_on_object])
                        task_command_type = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)
                        task_command_lang = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)
                    else:
                        task_command_type = []
                        task_command_lang = []
                        for stack_on_object_desc in config['object_description'][stack_on_object]:
                            for target_object_desc in config['object_description'][target_object]:
                                task_command_type.append(get_task_command(skill_desc, target_object_desc, stack_on_object_desc))
                                task_command_lang.append(get_task_command(skill_desc, target_object_desc, stack_on_object_desc))

                    env_config['task_command_lang'] = task_command_lang
                    env_config['task_command_type'] = task_command_type

                    target_object_color = target_object.split('_')[1]
                    stack_on_object_color = env_obj_config.stack_on_obj.split('_')[1]
                    env_config['task_command_color'] = get_task_command(skill_desc, f'{target_object_color} object', f'{stack_on_object_color} object')

                    env_configs.append(env_config.copy())

                elif ('put_in_open_drawer' in skill) or ('stick_door_close' in skill):

                    auxiliary_obj = auxiliary_objs[0]
                    auxiliary_obj_name = env_config['auxiliary_objects']

                    if sample_single_description:
                        auxiliary_object_desc = sample_description(config['object_description'][auxiliary_obj])
                        task_command_type = get_task_command(skill_desc, target_object_desc, auxiliary_object_desc)
                        task_command_lang = get_task_command(skill_desc, target_object_desc, auxiliary_object_desc)
                    else:
                        task_command_type = []
                        task_command_lang = []
                        for auxiliary_object_desc in config['object_description'][auxiliary_obj]:
                            for target_object_desc in config['object_description'][target_object]:
                                task_command_type.append(get_task_command(skill_desc, target_object_desc, auxiliary_object_desc))
                                task_command_lang.append(get_task_command(skill_desc, target_object_desc, auxiliary_object_desc))

                    env_config['task_command_lang'] = task_command_lang
                    env_config['task_command_type'] = task_command_type

                    env_configs.append(env_config.copy())
                elif skill == 'peg_insert':
                    assert 'RoundNut' in env_config['medium_objects'], 'No nut specified for peg insertion'
                    env_configs.append(env_config)

                else:
                    env_configs.append(env_config)

    return env_configs


def generate_multicolor_block_env_configs(config: DictConfig, skill_cfg: DictConfig):
    """
    For multicolor block env, each skill itself defines some meta-information for what
    kind of information does it want to generate.
    """
    skills = skill_cfg.all_skills if skill_cfg.use == 'all' else [skill_cfg.use]

    env_configs = []

    for skill in skills:
        only_use_block_objects = config.get('only_use_block_objects', False)
        num_demos_per_env = config[skill].num_demos_per_env

        if 'stack' in skill or ('pick_and_place' in skill and ('left' in skill or 'right' in skill or 'back' in skill or 'front' in skill)):
            target_objs = config[skill]['target_blocks']
            stack_on_objs = config[skill]['stack_on_objects']
            distractor_blocks = config['block_colors']['all']

            obj_configs = generate_stack_skill_obj_configs(
                target_objs, stack_on_objs, distractor_blocks,
                num_distractor_blocks=2,
                num_stack_on_objects_for_one_target=config[skill].num_stack_on_objects_for_one_target,
                num_distractor_block_variations=config[skill].num_distractor_block_variations)

        else:
            target_objs = config[skill]['target_blocks'] + config[skill]['target_objects']
            distractor_blocks = config['block_colors']['all']

            # Generate configs where the target object
            obj_configs = generate_single_target_obj_configs(
                config[skill]['target_blocks'], distractor_blocks,
                num_distractor_blocks=3,
                num_distractor_block_variations=config[skill].num_distractor_block_variations,
                is_target_object_a_block=True,)
            if not only_use_block_objects:
                obj_configs.extend(generate_single_target_obj_configs(
                    config[skill]['target_objects'], distractor_blocks,
                    num_distractor_blocks=4,
                    num_distractor_block_variations=config[skill].num_distractor_block_variations,
                    is_target_object_a_block=False,))

        for env_obj_config in obj_configs:

            block_size = 'medium'
            blockA_config = OrderedDict(size=block_size, color=env_obj_config.all_blocks[0])
            blockB_config = OrderedDict(size=block_size, color=env_obj_config.all_blocks[1])
            blockC_config = OrderedDict(size=block_size, color=env_obj_config.all_blocks[2])
            blockD_config = OrderedDict(size=block_size, color=env_obj_config.all_blocks[3])
            # Not specifying object_configs for now

            skill_desc = sample_description(config[skill]["skill_desc"])

            target_object = env_obj_config.target_obj
            target_object_desc = sample_description(config['object_description'][target_object])
            task_command_type = get_task_command(skill_desc, target_object_desc)
            task_command_lang = get_task_command(skill_desc, target_object_desc)

            if target_object.startswith('block_'):
                assert target_object == env_obj_config.all_blocks[0]
                target_object_color = blockA_config['color'].split('_')[1]
                target_object_size = blockA_config['size']

            env_config = OrderedDict(
                    # First block is always the target object in our case now
                    target_object='blockA' if target_object.startswith('block_') else target_object,
                    blockA_config=blockA_config,
                    blockB_config=blockB_config,
                    blockC_config=blockC_config,
                    blockD_config=blockD_config,
                    
                    task_command_color=get_task_command(skill_desc, f'{target_object_color} object'),
                    task_command_size=get_task_command(skill_desc, f'{target_object_size} object'),
                    
                    skill=skill,
                    # TODO: For now we set the skill as the task can change this later?
                    task=skill,
                    task_command_type=task_command_type,
                    task_command_lang=task_command_lang,

                    # Only use block objects?
                    only_use_block_objects=only_use_block_objects,
                    num_demos_per_env=num_demos_per_env,
                    update_block_colors=True,
                    all_blocks=copy.deepcopy(env_obj_config.all_blocks),
                    camera_names=list(config.get('camera_names', ['left_cap2'])),
                )
            if not only_use_block_objects:
                pass
            
            if ('stack' in skill) or ('pick_and_place' in skill and ('left' in skill or 'right' in skill or 'back' in skill or 'front' in skill)):
                # loop through which object to stack on
                assert env_obj_config.stack_on_obj is not None and target_object != env_obj_config.stack_on_obj

                stack_on_object = env_obj_config.stack_on_obj
                assert stack_on_object == env_obj_config.all_blocks[1]
                env_config['stack_on_object'] = 'blockB'

                stack_on_object_desc = sample_description(config['object_description'][stack_on_object])
                task_command_type = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)
                task_command_lang = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)
                env_config['task_command_lang'] = task_command_lang
                env_config['task_command_type'] = task_command_type

                stack_on_object_color = blockB_config['color'].split('_')[1]
                stack_on_object_size = blockB_config['size']
                env_config['task_command_color'] = get_task_command(skill_desc, f'{target_object_color} object', f'{stack_on_object_color} object')
                env_config['task_command_size'] = get_task_command(skill_desc, f'{target_object_size} object', f'{stack_on_object_size} object')

                env_configs.append(env_config.copy())

            elif skill == 'buttonpush':
                env_config['task_command_type'] = config[skill]["skill_desc"]
                env_config['task_command_lang'] = config[skill]["skill_desc_lang"]
                env_cfg_dict = recursively_get_dict_from_omegaconf(config[skill]['env_cfg'])
                env_config.update(env_cfg_dict)
                env_configs.append(env_config.copy())

            elif skill == 'binpick':
                task_command_type = skill_desc.replace('<target>', target_object_desc)
                task_command_type = task_command_type.replace('<color>', '')
                env_config['task_command_type'] = task_command_type
                task_command_lang = skill_desc.replace('<target>', target_object_desc)
                task_command_lang = task_command_lang.replace('<color>', 'light blue')
                env_config['task_command_lang'] = task_command_lang

                env_cfg_dict = recursively_get_dict_from_omegaconf(config[skill]['env_cfg'])
                env_config.update(env_cfg_dict)
                env_configs.append(env_config.copy())

            else:
                env_configs.append(env_config)

    return env_configs

def get_task_command(skill_desc, target, stack_on=''):
    if '<target>' in skill_desc:
        skill_desc = skill_desc.replace("<target>", target, 1)
    
    if '<object>' in skill_desc:
        skill_desc = skill_desc.replace("<object>", stack_on, 1)
    return skill_desc


def get_labelled_task_command_for_stack(skill_desc, target, stack_on):
    skill_desc = skill_desc.replace("<target>", target, 1)
    skill_desc = skill_desc.replace("<object>", stack_on, 1)
    return skill_desc


def sample_description(descriptions):
    if isinstance(descriptions, ListConfig):
        np.random.shuffle(descriptions)
        return descriptions[0]
    elif isinstance(descriptions, str):
        return descriptions
    else:
        raise NotImplementedError('Skill desc type not supported')

def generate_env_configs(config: DictConfig, skill_cfg: DictConfig):
    """
    For a certain skill, we sample over target objects for each env.
    With respect to env physical properties, we sample only locations of objects for now.
    TODO: Change object colors/textures
    TODO: Change object sizes
    """
    env_configs = []
    skills = skill_cfg.all_skills if skill_cfg.use == 'all' else [skill_cfg.use]

    for skill in skills:
        for env_objects in itertools.product(
            config[skill]['target_objects'],
            config.all_block_sizes,
            config.all_blockA_colors,
            config.all_blockB_colors,
            config.all_blockC_colors,
            config.all_blockD_colors,
            config.all_coke_colors,
            config.all_pepsi_colors,
            config.all_milk_colors,
            config.all_bread_colors,
            config.all_bottle_colors,):

            target_object = env_objects[0]
            block_size = env_objects[1]
            blockA_color = env_objects[2]
            blockB_color = env_objects[3]
            blockC_color = env_objects[4]
            blockD_color = env_objects[5]

            coke_color = env_objects[6]
            pepsi_color = env_objects[7]
            milk_color = env_objects[8]
            bread_color = env_objects[9]
            bottle_color = env_objects[10]

            blockA_config = OrderedDict(size=block_size, color=blockA_color)
            blockB_config = OrderedDict(size=block_size, color=blockB_color)
            blockC_config = OrderedDict(size=block_size, color=blockC_color)
            blockD_config = OrderedDict(size=block_size, color=blockD_color)
            coke_config = OrderedDict(size='medium', color=coke_color)
            pepsi_config = OrderedDict(size='medium', color=pepsi_color)
            milk_config = OrderedDict(size='medium', color=milk_color)
            bread_config = OrderedDict(size='medium', color=bread_color)
            bottle_config = OrderedDict(size='medium', color=bottle_color)

            skill_desc = sample_description(config[skill]["skill_desc"])

            target_object_desc = sample_description(config['object_description'][target_object])

            task_command_type = get_task_command(skill_desc, target_object_desc)
            task_command_lang = get_task_command(skill_desc, target_object_desc)

            target_object_config = eval(f'{target_object}_config')
            target_object_color = target_object_config['color']
            target_object_size = target_object_config['size']

            only_use_block_objects = config.get('only_use_block_objects', False)
            env_config = OrderedDict(
                    target_object=target_object,
                    blockA_config=blockA_config,
                    blockB_config=blockB_config,
                    blockC_config=blockC_config,
                    blockD_config=blockD_config,

                    # TODO: should be called red/green/blue/yellow block not blocksA/B/C/D
                    skill=skill,
                    # TODO: For now we set the skill as the task can change this later?
                    task=skill,
                    task_command_color= get_task_command(skill_desc, f'{target_object_color} object'),
                    task_command_size= get_task_command(skill_desc, f'{target_object_size} object'),
                    task_command_type=task_command_type,
                    task_command_lang=task_command_lang,

                    # Only use block objects?
                    only_use_block_objects=only_use_block_objects,
                )
            if not only_use_block_objects:
                env_config.update(dict(
                    coke_config=coke_config,
                    pepsi_config=pepsi_config,
                    milk_config=milk_config,
                    bread_config=bread_config,
                    bottle_config=bottle_config,
                ))
            
            if ('stack' in skill) or ('pick_and_place' in skill and ('left' in skill or 'right' in skill or 'back' in skill or 'front' in skill)):
                # loop through which object to stack on
                for stack_on_object in config[skill]['stack_on_objects']:
                    if stack_on_object == target_object: # can't stack on itself
                        continue
                    else:
                        stack_on_object_desc = sample_description(config['object_description'][stack_on_object])
                        task_command_type = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)
                        task_command_lang = get_task_command(skill_desc, target_object_desc, stack_on_object_desc)

                        env_config['stack_on_object'] = stack_on_object

                        stack_on_object_config = eval(f'{stack_on_object}_config')
                        stack_on_object_color = stack_on_object_config['color']
                        stack_on_object_size = stack_on_object_config['size']

                        env_config['task_command_color'] = get_task_command(skill_desc, f'{target_object_color} object', f'{stack_on_object_color} object')
                        env_config['task_command_size'] = get_task_command(skill_desc, f'{target_object_size} object', f'{stack_on_object_size} object')
                        env_config['task_command_type'] = task_command_type
                        env_config['task_command_lang'] = task_command_lang
                        env_configs.append(env_config.copy())

            elif skill == 'binpick':
                skill_desc = skill_desc.replace('<target>', target_object_desc)
                env_config['task_command_type'] = task_command_type

            else:
                env_configs.append(env_config)
    return env_configs

def get_env_name(env_config: DictConfig, data_config: DictConfig,
                 add_skill_name_in_env_name: bool = False, env_idx: int = 0):
    if env_config.get('update_block_colors', False):
        if 'door' in env_config['task'] or 'drawer' in env_config['task'] or 'window' in env_config['task']:
            target = env_config['task'] + '_' + f'target_{env_config["target_object"]}'
        elif 'faucet' in env_config['task']:
            target = env_config['task']
        elif 'peg_insert' in env_config['task'] or 'nut_pick' in env_config['task']:
            target = env_config['task']
        elif env_config['target_object'] in ('coke', 'pepsi'):
            target = env_config['target_object']
        elif env_config['target_object'].startswith('stick'):
            target = env_config['target_object']
        elif env_config['target_object'].startswith('block'):
            target = env_config['blockA_config']['color']
        # elif env_config.get('blocks'):
        #     target = env_config['blocks'][0]
        # elif env_config.get('all_blocks'):
        #     target = env_config['all_blocks'][0]
        else:
            target = env_config['target_object']
            # raise ValueError('Invalid target block not found.')
    else:
        target = env_config['target_object']

    if 'target' in data_config.vary:
        name = f'{data_config.vary}_{target}'
    elif 'lang' in data_config.vary:
        name = f'{data_config.vary}_{target}'
    elif 'color' in data_config.vary:
        color = env_config['block_config']['color']
        name = f'{data_config.vary}_{color}'
    else:
        raise NotImplementedError("Dataset variat not implemented.")
    
    if 'stack' in env_config['skill'] or ('pick_and_place' in env_config['skill'] and ('left' in env_config['skill'] or 'right' in env_config['skill'] or 'back' in env_config['skill'] or 'front' in env_config['skill'])):
        if env_config.get('update_block_colors', False):
            if env_config.get('blocks'):
                name = f'{name}_{env_config["blocks"][1]}'
            elif env_config.get('all_blocks'):
                name = f'{name}_{env_config["all_blocks"][1]}'
            else:
                raise ValueError('Invalid stack on block not found.')
        else:
            name = f'{name}_{env_config["stack_on_object"]}'

    if add_skill_name_in_env_name:
        name = f'env_{env_config["skill"]}_{name}'
    else:
        name = f'env_{name}'

    if env_config.get('distractor_cfg_key'):
        name = f'{name}_distractor_{env_config["distractor_cfg_key"]}'

    # We need an additional index at the end since we can have multiple environments
    # for the same target object (since we can have different distractor sets)
    if env_config.get('update_block_colors', False):
        name += f'_idx_{env_idx:02d}'

    return name


def collect_data(job_config: DictConfig, config: DictConfig, data_config: DictConfig,
                 skill_config: DictConfig, main_data_dir, data_type='train', num_demos=1):
    if job_config.get('multiobj', False):
        procedural = job_config.get('procedural', False)
        env_configs = generate_multiobj_multiskill_env_configs(config, skill_config, procedural=procedural)
    elif job_config.get('multicolor_blocks', False):
        env_configs = generate_multicolor_block_env_configs(config, skill_config)
    else:
        env_configs = generate_env_configs(config, skill_config)
    print(f'NUM OF {data_type} ENV CONFIGS: {len(env_configs)}')

    if data_config.save:
        # Save env configs
        to_save_env_configs = OrderedDict()
        data_dir = main_data_dir / data_type
        if not data_dir.exists():
            os.makedirs(data_dir)

        for env_idx, env_config in enumerate(env_configs):
            add_skill_name_in_env_name = skill_config.use == 'all'
            env_name = get_env_name(env_config, data_config,
                                    add_skill_name_in_env_name=add_skill_name_in_env_name,
                                    env_idx=env_idx)
            print(F"Collecting data for {data_type} env {env_idx}/{len(env_configs)}: {env_name}")
            num_demos_for_env = env_config.get('num_demos_per_env', num_demos)
            demos = collect_demo_data_for_env_config(env_config, num_demos_for_env)
            to_save_env_configs[env_name] = env_config
            save_demos(demos, data_dir, env_name, env_config.get('camera_names', ['left_cap2']))
        save_env_config(to_save_env_configs, main_data_dir, f'{data_type}_env_configs')


@hydra.main(version_base="1.1.0", config_name="sawyer_shape_gen_multiskill_procedural_with_bottles",config_path="config")
def main(config: DictConfig) -> None:

    np.random.seed(config['seed'])
    random.seed(config['seed'])

    if config.get('blocks'):
        config['blocks']['all'] = config['blocks']['train'] + config['blocks']['eval']
    # Long blocks are called sticks
    if config.get('sticks'):
        config['sticks']['all'] = config['sticks']['train'] + config['sticks']['eval']

    skill_name_short = config.skills.use.lower().replace(" ", "")
    if skill_name_short == 'all':
        dataset_name = f'{config.data.prefix}_multiskill_{config.data.vary}'
    else:
        dataset_name = f'{config.data.prefix}_{skill_name_short}_{config.data.vary}'
    print(f"Dataset name: {dataset_name}")

    main_data_dir = Path(config.data.data_dir) / dataset_name
    if not main_data_dir.exists():
        os.makedirs(main_data_dir)

    data_collect_types = config.get('data_collect_types', ['train', 'eval'])

    for data_type in data_collect_types:
        # Collect and save train data
        print(f"================Collecting data for {data_type} environments================")
        num_demos = config.data.num_demos_per_env_type[data_type]
        collect_data(config, config[data_type], config.data, config.skills[data_type], main_data_dir,
                    data_type=data_type, num_demos=num_demos)

    # Saving the config used for data generation
    # with open(main_data_dir/'data_cfg.yaml', "w") as f:
    #     OmegaConf.save(config, f)

if __name__ == "__main__":
    main()
